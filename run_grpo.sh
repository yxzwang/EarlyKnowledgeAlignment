# --- 1. 修改 getopts 以接收新的 -u (unique_id) 参数 ---
unique_id=""
while getopts "p:m:d:u:" opt; do
  case $opt in
    p) path=$OPTARG ;;
    m) model=$OPTARG ;;
    d) dataset=$OPTARG ;;
    u) unique_id=$OPTARG ;; # <--- 新增：接收唯一ID
    *) echo "Invalid option"; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

# --- 检查是否传入了唯一ID ---
if [ -z "$unique_id" ]; then
  echo "Error: -u (unique_id) is a required argument for isolated execution."
  exit 1
fi

# --- 2. 环境变量设置 (保持不变) ---
export VLLM_ATTENTION_BACKEND=XFORMERS
export BASE_MODEL="${path}"
export PROJECT_NAME='Graph-R1'
export EXPERIMENT_NAME="${model}_${dataset}_grpo_with_top_5_knowledge" # 这个默认值会被$@覆盖
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

# --- 3. ★★★ 隔离化的 Ray 环境设置 (现在位于此脚本内部) ★★★ ---
echo "[INFO in run_grpo.sh] Setting up isolated environment for ID: $unique_id"

# 使用传入的 unique_id 创建唯一的目录和符号链接
REAL_RAY_TEMP_DIR="ray_temp/Graph-R1/${unique_id}"
LINK_PATH=~/ray_temp_link_${unique_id}

# 使用 unique_id 的PID部分生成一个确定性的端口，避免随机性
# 提取 $$ 部分，例如从 "myhost_12345" 中提取 "12345"
PID_PART=${unique_id##*_} 
RAY_PORT=$((30000 + RANDOM % 20000))

echo "[INFO in run_grpo.sh] Ray temp dir: $REAL_RAY_TEMP_DIR"
echo "[INFO in run_grpo.sh] Ray port: $RAY_PORT"

# 清理并创建资源
rm -rf "$LINK_PATH"
mkdir -p "$REAL_RAY_TEMP_DIR"
ln -s "$REAL_RAY_TEMP_DIR" "$LINK_PATH"

# 使用隔离化的路径和端口启动 Ray
ray start --head --temp-dir "$LINK_PATH" --node-ip-address='127.0.0.1' --port="$RAY_PORT"

# 设置环境变量，让 Python 客户端连接到我们刚刚启动的、隔离的 Ray
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=datasets/"${dataset}"/processed_with_top_5_knowledge/train.parquet \
    data.val_files=datasets/"${dataset}"/processed_with_top_5_knowledge/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_start_length=4096 \
    data.max_tool_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    tool.env='search' $@

# --- 5. ★★★ 脚本结束时自动清理 ★★★ ---
# 启动 Ray 的脚本也应该负责停止它
echo "[INFO in run_grpo.sh] Task finished. Stopping Ray cluster on port $RAY_PORT."
ray stop