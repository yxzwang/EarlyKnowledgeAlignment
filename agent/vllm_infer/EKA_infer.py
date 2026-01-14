import argparse
import json
import re
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from agent.tool.tool_env import ToolEnv, step_batch
from agent.tool.tools import _default_tools

from eval import cal_em, cal_f1


# ---------------------- prompt & 工具调用相关 ---------------------- #

def build_system_prompt(question: str, initial_knowledge: List[str]) -> str:
    initial_knowledge_str = "\n".join(initial_knowledge) if initial_knowledge else ""

    instruction = (
        "Answer the given question.\n"
        "You must always first do your reasoning inside <think>...</think>.\n"
        "If you need external information, after finishing a <think> block you may call "
        "the search tool by outputting a <query>...</query> block that contains a JSON "
        "object with a single field \"query\".\n"
        "The environment will execute the search and return the top results wrapped in "
        "<knowledge>...</knowledge>. You can repeat this process and call the tool "
        "multiple times.\n"
        "When you have enough information to answer, stop calling the tool and output "
        "your final reasoning in <think>...</think> followed by the final answer in "
        "<answer>...</answer>.\n\n"
        "Output format for tool call:\n"
        "<think>\n"
        "...\n"
        "</think>\n"
        "<query>\n"
        "{\n"
        '  \"query\": \"...\"\n'
        "}\n"
        "</query>\n\n"
        "Output format for answer:\n"
        "<think>\n"
        "...\n"
        "</think>\n"
        "<answer>\n"
        "...\n"
        "</answer>\n"
    )

    system_content = (
        "<|im_start|>system\n"
        + instruction
        + f"\nQuestion: {question}\n"
    )

    if initial_knowledge_str:
        system_content += (
            "\nThe following initial knowledge may be useful. "
            "It is given to you before any tool calls and is wrapped in <knowledge> tags.\n"
            "<knowledge>\n"
            f"{initial_knowledge_str}\n"
            "</knowledge>\n"
        )

    system_content += "<|im_end|>\n<|im_start|>assistant\n"

    return system_content


def process_tool_call(responses_str: List[str]) -> Tuple[List[str], List[bool]]:
    """
    检查每条模型输出中是否包含 <query>...</query>：
    - 如果有：截断到 </query>，标记为 active=True（需要执行工具）
    - 不管有没有，都在末尾补上 <|im_end|>（vLLM chat 模板里的 end token）
    """
    def process_single_response(resp: str) -> Tuple[str, bool]:
        eos_token = "<|im_end|>"
        tool_call_end = "</query>"
        tool_pattern = r"<query>(.*?)</query>"
        match = re.search(tool_pattern, resp, re.DOTALL)

        if not match:
            return resp + eos_token, False

        resp = resp.split(tool_call_end)[0] + tool_call_end
        return resp + eos_token, True

    processed_strs = []
    active_masks = []
    for resp in responses_str:
        processed, active = process_single_response(resp)
        processed_strs.append(processed)
        active_masks.append(active)

    return processed_strs, active_masks


def execute_tool_calls_batch(
    response_strs: List[str],
    env: ToolEnv,
    active_masks: List[bool],
) -> List[str]:
    tool_custom_response_template = (
        "<|im_start|>user\n"
        "<knowledge>\n{tool_response}\n</knowledge>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )

    active_envs = []
    active_responses = []
    active_indices = []

    for i, (resp, active) in enumerate(zip(response_strs, active_masks)):
        if active:
            active_envs.append(env)
            active_responses.append(resp)
            active_indices.append(i)

    tool_responses: List[str] = [""] * len(response_strs)

    if not active_envs:
        return tool_responses

    batch_results = step_batch(active_envs, active_responses)

    for idx, result in zip(active_indices, batch_results):
        if result is None:
            tool_responses[idx] = ""
        else:
            tool_response = result[0]
            tool_responses[idx] = tool_custom_response_template.format(
                tool_response=tool_response
            )

    return tool_responses


def extract_answer_from_text(text: str) -> str:
    """
    从完整对话文本中抽取 <answer>...</answer> 的内容。
    找不到就尝试去掉前面的 <think>...</think>，实在不行就原样返回。
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    think_match = re.search(r"<think>.*?</think>\s*(.*)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    return text.strip()


def run_single_example(
    client: OpenAI,
    env: ToolEnv,
    model_name: str,
    question: str,
    initial_knowledge: List[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_turns: int,
) -> str:
    """
    对单条样本运行完整的“工具增强推理”流程，返回最终预测答案字符串。
    """
    system_message_content = build_system_prompt(question, initial_knowledge)

    # 和 run.py 一样，用一条 user message 包住整个模板
    messages = [{
        "role": "user",
        "content": system_message_content,
    }]

    final_response_text = ""

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        response_message = response.choices[0].message
        model_output = response_message.content or ""
        responses_str = [model_output]

        responses_str, active_masks = process_tool_call(responses_str)
        tool_responses = execute_tool_calls_batch(responses_str, env, active_masks)

        if active_masks[0] is True:
            # 这一轮发起了 <query>，把工具返回的 <knowledge> 接在后面继续下一轮
            prompt = messages[0]["content"] + responses_str[0] + tool_responses[0]
            messages = [{
                "role": "user",
                "content": prompt,
            }]
        else:
            # 没有 <query>，认为已经给出了最终 <answer>
            final_response_text = messages[0]["content"] + responses_str[0]
            break

    predicted_answer = extract_answer_from_text(final_response_text)
    return predicted_answer


# ---------------------- 数据加载 & 主流程 ---------------------- #

def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError("Dataset must be JSON array or {\"data\": [...]}")


def process_example_worker(
    idx: int,
    example: dict,
    args: argparse.Namespace,
) -> Tuple[int, str, str, List[str]]:
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )
    tools = _default_tools("search")
    env = ToolEnv(tools=tools, max_turns=args.max_turns)

    question = example["question"]
    golden_answers = example.get("golden_answers", [])
    initial_knowledge = example.get("initial_knowledge", [])

    prediction = run_single_example(
        client=client,
        env=env,
        model_name=args.model,
        question=question,
        initial_knowledge=initial_knowledge,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
    )

    return idx, prediction, question, golden_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the evaluation file for ONE dataset.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (used only for logging / saving).")

    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--api-base", type=str,
                        default="http://localhost:8002/v1")
    parser.add_argument("--model", type=str, default="agent")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=20)

    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of concurrent examples to run in parallel.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save per-example predictions as JSON.")

    args = parser.parse_args()

    dataset = load_dataset(args.data_path)
    num_examples = len(dataset)

    # 结果数组预先开好，保证 idx 对齐
    gold_answers_all: List[List[str]] = [None] * num_examples  # type: ignore
    pred_answers_all: List[str] = [None] * num_examples        # type: ignore
    results: List[dict] = [None] * num_examples                # type: ignore

    print(f"[INFO] Dataset: {args.dataset_name}, #examples = {num_examples}")
    print(f"[INFO] Using batch_size (concurrency) = {args.batch_size}")

    # 用线程池并行跑，每个线程一个 client + ToolEnv
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        future_to_idx = {}
        for idx, example in enumerate(dataset):
            fut = executor.submit(process_example_worker, idx, example, args)
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            idx, prediction, question, golden_answers = fut.result()

            gold_answers_all[idx] = golden_answers
            pred_answers_all[idx] = prediction
            results[idx] = {
                "index": idx,
                "question": question,
                "golden_answers": golden_answers,
                "prediction": prediction,
            }

            print(f"[{idx+1}/{num_examples}] Question: {question}")
            print(f"  Prediction: {prediction}")
            print(f"  Gold: {golden_answers}")
            print("-" * 60)

    # 计算 EM / F1（直接复用 eval.py）
    em = cal_em(gold_answers_all, pred_answers_all)
    f1 = cal_f1(gold_answers_all, pred_answers_all)

    print("=" * 80)
    print(f"Dataset: {args.dataset_name}")
    print(f"Number of examples: {num_examples}")
    print(f"Exact Match (EM): {em:.4f}")
    print(f"F1: {f1:.4f}")
    print("=" * 80)

    if args.output_path:
        output = {
            "dataset_name": args.dataset_name,
            "num_examples": num_examples,
            "em": em,
            "f1": f1,
            "examples": results,
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
