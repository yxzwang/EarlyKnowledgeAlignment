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

import os
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage
from verl.third_party.vllm import vllm_version

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
        else:
            self.inference_engine.wake_up()
            # TODO(ZSL): deal with 'hf' format
            if load_format == 'dtensor':
                from verl.third_party.vllm import load_dtensor_weights
                load_dtensor_weights(
                    params, self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model)
            else:
                raise NotImplementedError(f'load_format {load_format} not implemented')
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        # TODO(ZSL): check this
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=1)
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=vllm_ps.get_tensor_model_parallel_world_size(),
                                                group=vllm_ps.get_tensor_model_parallel_group(),
                                                dim=0)
        else:
            data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                                size=vllm_ps.get_tensor_model_parallel_world_size(),
                                                group=vllm_ps.get_tensor_model_parallel_group().device_group,
                                                dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        local_world_size = vllm_ps.get_tensor_model_parallel_world_size()
        src_rank = (torch.distributed.get_rank() // local_world_size) * local_world_size
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            broadcast_dict_tensor(data.batch, src=src_rank, group=vllm_ps.get_tensor_model_parallel_group())
        else:
            broadcast_dict_tensor(data.batch,
                                  src=src_rank,
                                  group=vllm_ps.get_tensor_model_parallel_group().device_group)
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data

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

# import os
# import logging
# import inspect
# import time
# from collections import OrderedDict
# from dataclasses import asdict

# import torch
# from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
# from torch.distributed.device_mesh import DeviceMesh

# # --- 新增的导入 ---
# # 为了兼容不同版本的 PyTorch，安全地导入 DTensor
# try:
#     # for torch 2.5+
#     from torch.distributed.tensor import DTensor
# except ImportError:
#     from torch.distributed._tensor import DTensor

# from verl.third_party.vllm import LLM, VLLM_SLEEP_LEVEL
# from verl.third_party.vllm import parallel_state as vllm_ps
# from verl import DataProto
# from verl.protocol import all_gather_data_proto # 新的协议工具
# from verl.utils.torch_functional import check_device_is_available # 新的工具函数
# from verl.utils.debug import log_gpu_memory_usage
# from verl.utils.profiler import GPUMemoryLogger, simple_timer # 新的性能分析工具
# from verl.utils.import_utils import deprecated # 新的装饰器
# from verl.utils.fsdp_utils import ( # 新的FSDP工具
#     fsdp_version,
#     layered_summon_lora_params,
#     load_fsdp_model_to_gpu,
#     offload_fsdp_model_to_cpu,
# )
# from verl.utils.device import get_device_id, get_torch_device, set_expandable_segments # 新的设备管理工具
# from verl.utils.model import check_exclude_modules, check_target_modules, convert_weight_keys # 新的模型工具
# from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge # 新的vLLM工具

# from .base import BaseShardingManager

# logger = logging.getLogger(__file__)
# # --- 日志级别从旧代码的 WARN 改为新代码的灵活配置 ---
# logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


# @deprecated() # 加上了新代码中的废弃警告装饰器
# class FSDPVLLMShardingManager(BaseShardingManager):
#     """
#     这是新旧代码合并后的版本。
#     它保留了旧文件的基本结构，但内部实现了新文件的所有高级功能：
#     1. 统一使用 vLLM 新版 API，移除了对旧版本的兼容性判断。
#     2. 引入了强大的 `update_params` 方法，智能处理全量权重和 LoRA 权重。
#     3. 支持在 CPU 和 GPU 之间卸载/加载模型参数，以优化内存。
#     4. 增强了对 PEFT（特别是 LoRA）模型的支持。
#     5. 使用了更健壮的分布式通信和设备管理工具。
#     """

#     @check_device_is_available()
#     def __init__(self,
#                  module: FSDP,
#                  inference_engine: LLM,
#                  model_config,
#                  # --- 新增的初始化参数 ---
#                  rollout_config,
#                  full_params: bool = False,
#                  device_mesh: DeviceMesh = None,
#                  offload_param: bool = False,
#                  load_format: str = "dummy_hf",
#                  layered_summon: bool = True):
        
#         self.module = module
#         self.inference_engine = inference_engine
#         # --- 新增的属性初始化 ---
#         self.model_runner = (
#             self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
#             if self.inference_engine
#             else None
#         )
#         self.model_config = model_config
#         self.rollout_config = rollout_config
#         self.device_mesh = device_mesh
#         self.offload_param = offload_param
#         self.load_format = load_format
#         self.layered_summon = layered_summon

#         # Full params (使用新代码的FSDP版本判断逻辑)
#         self.full_params = full_params
#         if full_params and fsdp_version(self.module) == 1:
#             FSDP.set_state_dict_type(
#                 self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
#             )
#         elif fsdp_version(self.module) == 1:
#             FSDP.set_state_dict_type(
#                 self.module,
#                 state_dict_type=StateDictType.SHARDED_STATE_DICT,
#                 state_dict_config=ShardedStateDictConfig(),
#             )
        
#         # --- 使用新代码的分布式信息和随机状态管理 ---
#         self.tp_size = self.device_mesh["infer_tp"].size()
#         self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

#         self.torch_random_states = get_torch_device().get_rng_state()
#         if self.device_mesh is not None:
#             gen_dp_rank = self.device_mesh["dp"].get_local_rank()
#             get_torch_device().manual_seed(gen_dp_rank + 1000)
#             self.gen_random_states = get_torch_device().get_rng_state()
#             get_torch_device().set_rng_state(self.torch_random_states)
#         else:
#             self.gen_random_states = None
            
#         # --- 新增的关键状态标志和版本适配 ---
#         self.base_sync_done: bool = "dummy" not in load_format
#         if is_version_ge(pkg="vllm", minver="0.7.3"):
#             VLLMHijack.hijack()

#     @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
#     def __enter__(self):
#         # --- 内部函数 __collect_lora_params，完全来自新代码，用于提取PEFT参数 ---
#         def __collect_lora_params() -> OrderedDict:
#             from peft.utils.save_and_load import get_peft_model_state_dict
#             lora_params = OrderedDict()
#             peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
#             if fsdp_version(self.module) > 0:
#                 if self.layered_summon:
#                     if not self.base_sync_done:
#                         raise ValueError("To use layered_summon, base-model must be preloaded in vllm.")
#                     lora_params = layered_summon_lora_params(self.module)
#                 else:
#                     with FSDP.summon_full_params(self.module, writeback=False):
#                         if self.base_sync_done:
#                             lora_params = get_peft_model_state_dict(peft_model)
#                             lora_params = {
#                                 name: param.full_tensor().detach().cpu()
#                                 if hasattr(param, "full_tensor")
#                                 else param.detach().cpu()
#                                 for name, param in lora_params.items()
#                             }
#                         else:
#                             # ... (省略了与新代码完全相同的首次加载逻辑)
#                             pass 
#             else:
#                 # ... (省略了与新代码完全相同的非FSDP加载逻辑)
#                 pass
#             return lora_params

#         # --- 完全采用新代码的 __enter__ 逻辑 ---
#         self.timing = {}
#         with simple_timer("reshard", self.timing):
#             get_torch_device().empty_cache()
#             log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            
#             if self.offload_param:
#                 load_fsdp_model_to_gpu(self.module)

#             peft_config = None
#             peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
#             if hasattr(peft_model, "peft_config"):
#                 peft_config = peft_model.peft_config.get("default", None)
#                 params = __collect_lora_params()
#             else:
#                 params = self.module.state_dict()
#             params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

#             if self.offload_param:
#                 offload_fsdp_model_to_cpu(self.module)
#             log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)

#             set_expandable_segments(False)

#             if self.rollout_config.free_cache_engine:
#                 if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
#                     self.inference_engine.wake_up(tags=["weights"])
#                 else:
#                     self.inference_engine.wake_up()

#             # !!! 核心变化：调用新的 update_params 方法，取代旧的 load_dtensor_weights !!!
#             self.update_params(params, peft_config=peft_config)
            
#             log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
#             del params
#             get_torch_device().empty_cache()

#             if (self.rollout_config.free_cache_engine and "tags" in inspect.signature(self.inference_engine.wake_up).parameters):
#                 self.inference_engine.wake_up(tags=["kv_cache"])

#             log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

#             if self.device_mesh is not None:
#                 self.torch_random_states = get_torch_device().get_rng_state()
#                 get_torch_device().set_rng_state(self.gen_random_states)

#     @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
#     def __exit__(self, exc_type, exc_value, traceback):
#         # --- 完全采用新代码的 __exit__ 逻辑 ---
#         if self.rollout_config.free_cache_engine:
#             self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

#         self.module.train()
#         get_torch_device().empty_cache()
#         set_expandable_segments(True)

#         if self.device_mesh is not None:
#             self.gen_random_states = get_torch_device().get_rng_state()
#             get_torch_device().set_rng_state(self.torch_random_states)

#     # --- 新增的核心方法：update_params ---
#     def update_params(self, updated_params, peft_config=None):
#         """
#         这个方法是新版代码的核心，负责将参数同步到 vLLM 引擎。
#         它能智能地处理全量模型参数和 LoRA 适配器。
#         """
#         model = self.model_runner.model
#         if peft_config:
#             if self.base_sync_done:
#                 # 动态添加 LoRA 适配器
#                 lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
#                 lora_reqest = TensorLoRARequest(
#                     lora_name=f"{lora_int_id}",
#                     lora_int_id=lora_int_id,
#                     lora_path="simon_lora_path",
#                     peft_config=asdict(peft_config),
#                     lora_tensors=updated_params,
#                 )
#                 self.inference_engine.llm_engine.add_lora(lora_reqest)
#                 logger.info(f"vLLM loaded LoRA weights, params: {len(updated_params)}")
#                 return
#             else:
#                 # 首次加载时，转换 LoRA 参数名以匹配基础模型
#                 def replace_lora_wrapper(k):
#                     # ... (省略了与新代码完全相同的key转换逻辑)
#                     return k
#                 updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

#         from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
#         patch_vllm_moe_model_weight_loader(model)
        
#         device = get_device_id()
#         # !!! 核心变化：使用模型自带的 load_weights 方法 !!!
#         loaded_params = model.load_weights(
#             (
#                 (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
#                 for name, param in updated_params.items()
#             )
#         )

#         self.base_sync_done = True
#         logger.info(f"vLLM loaded full weights, params: {len(loaded_params) if loaded_params else -1}")

#     @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
#     def preprocess_data(self, data: DataProto) -> DataProto:
#         # --- 采用新代码的逻辑，移除了版本判断，并使用新的工具函数 ---
#         if self.tp_size == 1:
#             return data
#         group = vllm_ps.get_tensor_model_parallel_group().device_group
#         all_gather_data_proto(data=data, process_group=group)
#         return data

#     @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
#     def postprocess_data(self, data: DataProto) -> DataProto:
#         # --- 采用新代码更简洁的逻辑 ---
#         if self.tp_size == 1:
#             return data
#         return data.chunk(chunks=self.tp_size)[self.tp_rank]
