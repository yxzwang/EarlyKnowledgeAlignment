import json
import os
from eval_r import cal_rsim
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import traceback
import glob
import re
# import re

def remove_second_knowledge(text):
    # 匹配所有<knowledge>...</knowledge>标签对
    pattern = r'<knowledge>.*?</knowledge>'
    matches = list(re.finditer(pattern, text))
    
    if len(matches) >= 2:
        # 获取第二个匹配项的位置
        second_match = matches[1]
        # 删除第二个匹配项
        result = text[:second_match.start()] + text[second_match.end():]
        return result
    else:
        return text

def calculate_rsim_for_sample(d):
    try:
        # 去重 context
        context = []
        for c in d['context']:
            if c not in context:
                context.append(c)

        rsim_score = cal_rsim(["\n".join(context)], [d['knowledge']]) if d['knowledge'] != "" else 0.0 
        return rsim_score
    except Exception as e:
        print(f"[ERROR] Failed processing sample: {d.get('question', 'N/A')}")
        traceback.print_exc()
        return 0.0

def extract_step_number(filename):
    """从文件名中提取step数字"""
    match = re.search(r'results_step(\d+)\.json', filename)
    return int(match.group(1)) if match else None

def calculate_rsim_for_dir(args):
    dir_path = args.dir
    
    try:
        # 查找所有results_step*.json文件
        results_files = glob.glob(os.path.join(dir_path, "results_step*.json"))
        
        if not results_files:
            print(f"[WARNING] No results_step*.json files found in {dir_path}")
            return
        
        print(f"[INFO] Found {len(results_files)} results files to process")
        
        for results_file in tqdm(results_files, desc="Processing files"):
            step_num = extract_step_number(os.path.basename(results_file))
            if step_num is None:
                print(f"[WARNING] Could not extract step number from {results_file}")
                continue
                
            evals_file = os.path.join(dir_path, f"evals_step{step_num}.json")
            
            # 检查对应的evals文件是否存在
            if not os.path.exists(evals_file):
                print(f"[WARNING] Corresponding evals file not found: {evals_file}")
                continue
            
            try:
                # 读取results文件
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                # 准备数据进行rsim计算
                data_for_rsim = []
                for x in results_data:
                    # 去重 context
                    context = []
                    for c in x['context']:
                        if c not in context:
                            context.append(c)
                    # new_prediction = remove_second_knowledge(x['prediction'])
                    # 提取knowledge
                    knowledge = []
                    ksplit = x['prediction'].split("</knowledge>")[:-1]
                    for k in ksplit:
                        knowledge.append('<knowledge>'.join(k.split("<knowledge>")[1:]))
                    knowledge = '\n'.join(knowledge)
                    
                    data_for_rsim.append({
                        'context': context,
                        'knowledge': knowledge
                    })
                
                # 并行计算rsim分数
                max_workers = 16
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    rsim_scores = list(tqdm(
                        executor.map(calculate_rsim_for_sample, data_for_rsim), 
                        total=len(data_for_rsim), 
                        desc=f"Calculating R-Sim for step {step_num}",
                        leave=False
                    ))
                
                # 计算平均rsim分数
                avg_rsim = sum(rsim_scores) / len(rsim_scores) if rsim_scores else 0.0
                
                # 读取现有的evals文件
                with open(evals_file, 'r', encoding='utf-8') as f:
                    evals_data = json.load(f)
                
                # 添加rsim分数
                evals_data["val/answer_rsim_score_1/2WikiMultiHopQA"] = avg_rsim
                
                # 保存更新后的evals文件
                with open(evals_file, 'w', encoding='utf-8') as f:
                    json.dump(evals_data, f, indent=4)
                
                print(f"[SUCCESS] Updated {evals_file} with R-Sim score: {avg_rsim:.4f}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process {results_file}: {str(e)}")
                traceback.print_exc()
                continue
        
        print(f"[COMPLETED] Finished processing all files in {dir_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to process directory {dir_path}: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dir', type=str, default='../expr_results/entropy-Qwen2.5-7B-Instruct_2WikiMultiHopQA_lr_5e-7_grpo_with_top_5_knowledge')

    args = parse.parse_args()
    calculate_rsim_for_dir(args)