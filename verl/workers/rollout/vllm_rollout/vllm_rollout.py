# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = int(self.config.get('max_num_batched_tokens', 8192))

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )
        # kwargs = dict(
        #     n=1,
        #     logprobs=tokenizer.vocab_size, # <--- 修改这里！
        #     max_tokens=config.response_length,
        # )
        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # Add custom stop words if provided in the config
        if 'custom_stop_words' in config and len(config.custom_stop_words) > 0:
            kwargs['detokenize'] = True
            kwargs['stop'] = config.custom_stop_words

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    # # 文件: vllm_rollout.py
    # # 在 vLLMRollout 类定义内部，可以放在 generate_sequences 函数前面或后面

    # # ★★★ 新增辅助函数：处理 vLLM 的 logprobs 输出 ★★★
    # def _process_logprobs_to_tensor(self, logprobs_list: List, max_len: int, device: torch.device) -> torch.Tensor:
    #     """
    #     一个辅助函数，用于将 vLLM 的 logprobs 格式转换为一个密集的 logits 张量。
    #     vLLM 的输出格式是 List[List[Dict[int, float]]]。
    #     """
    #     batch_size = len(logprobs_list)
    #     vocab_size = self.inference_engine.tokenizer.vocab_size
        
    #     # 初始化一个用-inf填充的张量，代表极低的概率
    #     logits_tensor = torch.full((batch_size, max_len, vocab_size), float('-inf'), device=device)
        
    #     for i, seq_logprobs in enumerate(logprobs_list):
    #         if seq_logprobs is None:
    #             continue
    #         seq_len = len(seq_logprobs)
    #         for t in range(seq_len):
    #             step_logprobs = seq_logprobs[t]
    #             if step_logprobs:
    #                 # vLLM 返回的是一个字典 {token_id: logprob}
    #                 # 我们需要把它填回张量的正确位置
    #                 token_ids = list(step_logprobs.keys())
    #                 log_probs = list(step_logprobs.values())
                    
    #                 # 使用 advanced indexing 快速填充
    #                 logits_tensor[i, t, token_ids] = torch.tensor(log_probs, device=device)
                    
    #     return logits_tensor

    # @torch.no_grad()
    # def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    #     # rebuild vllm cache engine
    #     if self.config.free_cache_engine:
    #         self.inference_engine.init_cache_engine()

    #     idx = prompts.batch['input_ids']  # (bs, prompt_length)
    #     # left-padded attention_mask
    #     attention_mask = prompts.batch['attention_mask']
    #     position_ids = prompts.batch['position_ids']

    #     # used to construct attention_mask
    #     eos_token_id = prompts.meta_info['eos_token_id']

    #     batch_size = idx.size(0)

    #     idx_list = []
    #     # parse idx from torch.Tensor to List[List[str]]
    #     for i in range(batch_size):
    #         idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

    #     do_sample = prompts.meta_info.get('do_sample', True)
    #     if not do_sample:
    #         kwargs = {
    #             'best_of': 1,
    #             'top_p': 1.0,
    #             'top_k': -1,
    #             'min_p': 0.0,
    #             'temperature': 0,
    #             'n': 1  # if greedy, only 1 response
    #         }

    #     # users can customize different sampling_params at different run
    #     with self.update_sampling_params(**kwargs):
    #         # output 现在是 List[RequestOutput]
    #         output = self.inference_engine.generate(
    #             prompts=None,  # because we have already convert it to prompt token id
    #             sampling_params=self.sampling_params,
    #             prompt_token_ids=idx_list,
    #             use_tqdm=False)

    #     # ★★★ 核心修改：处理新的输出格式 (List[RequestOutput]) ★★★
    #     # 从输出对象中分别提取 token_ids 和 logprobs
    #     response_ids_list = [out.outputs[0].token_ids for out in output]
    #     log_probs_list = [out.outputs[0].logprobs for out in output]

    #     # 手动对 response 进行 padding，使其成为一个 tensor
    #     response = [torch.tensor(r, device=idx.device) for r in response_ids_list]
    #     response = torch.nn.utils.rnn.pad_sequence(response, batch_first=True, padding_value=self.pad_token_id)

    #     # 如果生成的序列长度小于配置的长度，则填充到目标长度
    #     if response.shape[1] < self.config.response_length:
    #         response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)

    #     # ★★★ 核心修改：调用新辅助函数，将 log_probs_list 转换为密集的 logits 张量 ★★★
    #     logits = self._process_logprobs_to_tensor(log_probs_list, response.shape[1], idx.device)

    #     if self.config.n > 1 and do_sample:
    #         idx = idx.repeat_interleave(self.config.n, dim=0)
    #         attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
    #         position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
    #         batch_size = batch_size * self.config.n
    #     seq = torch.cat([idx, response], dim=-1)

    #     response_length = response.size(1)
    #     delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
    #     delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

    #     response_position_ids = position_ids[:, -1:] + delta_position_id
    #     position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    #     response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
    #     attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

    #     # ★★★ 核心修改：将 logits 加入返回的 TensorDict ★★★
    #     batch = TensorDict(
    #         {
    #             'prompts': idx,
    #             'responses': response,
    #             'logits': logits,  # <--- 在这里加入 logits！
    #             'input_ids': seq,  # here input_ids become the whole sentences
    #             'attention_mask': attention_mask,
    #             'position_ids': position_ids
    #         },
    #         batch_size=batch_size)

    #     # free vllm cache engine
    #     if self.config.free_cache_engine:
    #         self.inference_engine.free_cache_engine()

    #     return DataProto(batch=batch)


    # original
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
