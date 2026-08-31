# NPU Qwen3-32B GSPO Optimization Practice

Last updated: 07/03/2026.

To ensure a smooth experience, switch verl to the main branch, with commit ID 9d05508f5e3bd8ecb70cf94ab10dc087b57a716d. Note that the main branch may cause patch issues due to iterative refactoring; for a stable version, switch to `release/v0.8.0`.

The training script for this practice is: [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh) (`examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh`).

## Algorithm Adaptation

GSPO improves training stability by raising the optimization granularity from the **token level** to the **sequence level**, which avoids the **sharp increase in variance** that GRPO encounters and that leads to unstable training. At the same time, this algorithm also improves the convergence speed to a certain extent.

To successfully invoke the GSPO algorithm in the verl repository, you must complete the following required configuration.

# Core Algorithm Configuration
algorithm.adv_estimator=grpo \                    # Use the GRPO advantage estimator
algorithm.use_kl_in_reward=False \                # Do not add KL penalty to the reward
# GSPO Policy Loss Mode
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \ # Enable GSPO policy loss
# Extremely Small Clipping Range (GSPO Feature)
actor_rollout_ref.actor.clip_ratio_low=0.0003 \   # Lower clipping bound, value recommended in the paper
actor_rollout_ref.actor.clip_ratio_high=0.0004 \  # Upper clipping bound, value recommended in the paper
# KL Configuration (GSPO does not use KL loss)
actor_rollout_ref.actor.use_kl_loss=False \       # Disable KL loss
actor_rollout_ref.actor.kl_loss_coef=0.0 \        # Set the KL loss coefficient to 0
# Sequence-Level Loss Aggregation Mode (Core of GSPO)
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \ # Sequence-level averaging, recommended by the GSPO paper
# Batch Configuration
actor_rollout_ref.rollout.n=16 \                  # Generate 16 responses per prompt (group sampling)
```

Generally, the entry function is set to `verl.trainer.main_ppo`. For a complete runnable example, see [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh).

## Basic Environment

Currently, Atlas 800T A3 and Atlas 900 A3 SuperPoD are supported. Completing this best practice requires 4 Atlas 800T A3 devices.

### Install the Base Environment

| software      | version                                                    |
| ------------- | ---------------------------------------------------------- |
| Python        | 3.11                                                       |
| CANN          | ==9.0.0.B160 (CANN900B160)                                 |
| torch         | ==2.9.0                                                    |
| torch_npu     | ==2.9.0                                                    |
| triton_ascend | ==3.2.1                                                    |
| verl          | main                                                       |
| vllm          | v0.18.0                                                    |
| vllm-ascend   | v0.18.0                                                    |
| transformers  | 5.3.0                                                      |


```bash
cd verl
git checkout main
# Specify the corresponding recipe version
git submodule update --init --recursive recipe
```

### Obtaining Weights

Download the corresponding model weights from the Hugging Face library: [Qwen/Qwen3-32B · Hugging Face](https://huggingface.co/Qwen/Qwen3-32B)

### Dataset Preparation

```bash
# Download the math-17k dataset
git clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k

# Download the AIME_2024 test dataset
git clone https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
```

### Installing jemalloc

To ensure that Ray processes can properly reclaim memory, install and enable the jemalloc library for memory management.

#### Ubuntu operating system

Install jemalloc through the operating system source (note: the Ubuntu version must be 20.04 or later):

```shell
sudo apt install libjemalloc2
```

Before starting the task, run the following command to import jemalloc through an environment variable. First, confirm that the file exists by running **find /usr -name libjemalloc.so.2**:

```shell
# arm64 architecture
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64 architecture
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

#### OpenEuler operating system

Run the following command to install jemalloc through the operating system source.

```shell
yum install jemalloc
```

If the methods above do not work for installation, you can install through source code compilation. Go to the jemalloc official website to download the latest stable version. The official website address is: https://github.com/jemalloc/jemalloc/releases/

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

Before running the task, set the environment variable to import jemalloc by executing the following command:

```shell
# Set the environment variable based on the actual installation path. For example, if the installation path is /usr/local/lib/libjemalloc.so.2, you can set the environment variable using the following command (you can verify that the file exists by running find /usr -name libjemalloc.so.2).
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

### Launching Multi-Node Tasks

For the multi-machine tasks provided in this practice, you can use the following script to launch them.

```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2

# Modify this to the path of the test case you want to run
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

NNODES=4
NPUS_PER_NODE=16
# Modify this to the IP address of the master node
MASTER_ADDR="IP FOR MASTER NODE"
# Modify this to the communication network interface of the current node
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# Get the current IP address
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # Start the master node
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # Check whether device_count is equal to NNODES
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # Child nodes attempt to register with the Ray cluster until successful
  while true; do
      # Attempt to connect to the Ray cluster
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # Check whether the connection is successful
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600
```

DEFAULT_SH: Set this to the path of the configuration sh file used for training. In this case, set it to [run_qwen3_32b_fsdp.sh](../../../../../examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh) (that is, `examples/ascend_extras/gspo_trainer/run_qwen3_32b_fsdp.sh`).

NNODES and NPUS_PER_NODE: Set these to the number of nodes and the number of NPUs per node, respectively. In this case, they are 4 and 16.

MASTER_ADDR: Change it to the IP address of the corresponding master node. That is, the MASTER_ADDR of all nodes should be the same.

SOCKET_IFNAME, HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME: Set them to the corresponding communication network interface. You can obtain the communication network interface by running the following command:

```bash
ifconfig |grep "$(hostname -I |awk '{print $1}'|awk -F '.' '{print $0}')" -B 1|awk -F ':' '{print$1}' | head -1 | tail -1
```

## Performance Tuning

Optimization starts with four aspects: training, inference, scheduling, and others.

### Training

#### Dynamic batch size

```bash
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
```

**This optimization mainly adjusts the two parameters above. However, note that setting these two parameters too large can cause out-of-memory (OOM) errors.**

**Key adjustments** Increasing `actor_ppo_max_token_len` reduces training time, while adjusting `infer_ppo_max_token_len` provides no significant benefit and can be left unchanged.

**The following describes the functions of these two parameters:**

**These two parameters control the maximum number of tokens processed by each GPU in dynamic batch size mode.**

- **`actor_ppo_max_token_len`**: The maximum number of tokens that each GPU can process during the PPO update (forward and backward propagation) of the Actor model.
- **`infer_ppo_max_token_len`**: The maximum number of tokens that each GPU can process when calculating log probabilities during the inference phase (Reference policy and Rollout).

### Inference

#### ACLgraph+FULL_DECODE_ONLY

Optimization of inference operator dispatch can yield an average performance gain of approximately `15%~20%`.

Let’s first look at enabling **ACLgraph** alone, as follows:

```bash
# Enable ACLgraph+FULL_DECODE_ONLY (Note: When this parameter is set to False, TASK_QUEUE_ENABLE must be set to 1; otherwise, an error is reported)
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[8,16,32,64,128]' \
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY'
```

After `FULL_DECODE_ONLY` is enabled successfully, the following output appears:

![FULL_DECODE_ONLY result](https://github.com/wucong25/verl-data/blob/main/ascend_acl_graph.png)

**Guidelines for Setting the `cudagraph_capture_sizes` Parameter**

The value set in `cudagraph_capture_sizes` corresponds to the batch size. Here, the batch size is not the batch size corresponding to the DP domain in the configuration; it is the batch size relative to vLLM, with the unit being **tokens**.

The default generated algorithm is as follows, for reference only.

![cudagraph_capture_sizes](https://github.com/wucong25/verl-data/blob/main/ascend_set_cudagraph_sizes.png)

##### Switching the Inference Backend

Usage: `export VLLM_ATTENTION_BACKEND=XFORMERS`

![VLLM_ATTENTION_BACKEND](https://github.com/wucong25/verl-data/blob/main/ascend_vllm_attn_backend.png)

Note: Some backends are not supported in certain older versions of vllm-ascend.

##### Enable vLLM v1

Usage: `export VLLM_USE_V1=1`

It is generally beneficial to keep it enabled at all times.

### Scheduling

#### AIV

Open the method: Set `export HCCL_OP_EXPANSION_MODE="AIV"`

HCCL_OP_EXPANSION_MODE is an environment variable used to configure where communication algorithm orchestration expansion occurs. It supports the following values:

- AI_CPU: Indicates that the orchestration expansion of the communication algorithm is located on the AI CPU computing unit on the Device side.
- AIV: Indicates that the orchestration expansion of the communication algorithm is located on the Vector Core computing unit on the Device side.
- HOST: Indicates that the orchestration expansion of the communication algorithm is located on the Host-side CPU, and the Device side automatically selects the appropriate scheduler based on the hardware model.
- HOST_TS: Indicates that the orchestration expansion of the communication algorithm is located on the Host-side CPU. The Host sends tasks to the Device's Task Scheduler, which then schedules and executes the tasks.

The following describes two expansion mechanisms.

##### HOST expansion

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue1.png" alt="image-20260113194257095" style="zoom:50%;" />

- The software stack runs on the host CPU, and the communication algorithm expands into individual tasks.
- Each task calls the runtime interface and dispatches it to the device's rtsqueue.
- STARS sequentially retrieves tasks from the rtsqueue.
- Based on the task type, it invokes the SDMA and RDMA engines respectively.
  **Single operator bottleneck**: hostbound task submission takes 2 to 5 microseconds per task, and a communication operator involves hundreds of tasks. In single-operator scenarios, tasks are not cached on the device; each task is dispatched and executed one at a time.

##### AICPU mechanism expansion

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_task_queue3.png" alt="image-20260113194333218" style="zoom:50%;" />

- On the host side, tasks are not dispatched one by one. Instead, communication operators are treated as individual kernels and placed on the communication operator kernel queue.
- STARS schedules the kernels on the kernel queue stream and places them on the AiCPU for execution.
- AICPU calls the function (kernel), executes the kernel function with a single thread, expands the communication tasks within the function, and places the tasks on the rtsqueue for STARS to call.
- This reduces host-AiCPU interactions from hundreds of times to a single time.
- Task submission occurs on the AICPU, with partial merging of the submission operations.

#### TASK_QUEUE_ENABLE

**Usage:** `export TASK_QUEUE_ENABLE=2`

TASK_QUEUE_ENABLE, dispatch optimization, set the graph mode to 1 (that is, set it to 1 when graph mode is enabled), and set it to 2 for non-graph mode.

Diagram:

![ascend task queue](https://github.com/wucong25/verl-data/blob/main/ascend_task_queue2.png)

##### Bind-core Optimization

**Usage:** `export CPU_AFFINITY_CONF=1`

For detailed configuration principles, see: https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0059.html

### Other

The following content summarizes the tuning configurations for several global environment variables. Because these parameters often bring positive benefits in both the training and inference phases, and there is currently a lack of sufficiently detailed ablation experiments to strictly distinguish their respective contributions to training or inference, they are consolidated here for ongoing monitoring and further breakdown analysis.

#### Enable jemalloc

Usage (note that the jemalloc library must be installed first): `export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2`

**Installation and Usage Guide:** [MindSpeed-RL/docs/install_guide.md · Ascend/MindSpeed-RL - AtomGit | GitCode](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/install_guide.md#高性能内存库-jemalloc-安装)

#### Multi-Stream Multiplexing

Memory optimization is applied.

Enablement method: `export MULTI_STREAM_MEMORY_REUSE=1`

Principle introduction: https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0040.html

#### VLLM_ASCEND_ENABLE_FLASHCOMM

Usage: `export VLLM_ASCEND_ENABLE_FLASHCOMM=1`

Enable the FLASHCOMM high-speed communication optimization technology specific to Ascend NPU.

Address: https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

#### VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE

Usage: `export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1`

Enable dense computation optimization for large model inference on Ascend NPU

Address: https://vllm-ascend.readthedocs.io/zh-cn/latest/user_guide/release_notes.html

#### VLLM_ASCEND_ENABLE_PREFETCH_MLP

Usage: `export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1`

Enable the weight prefetch mechanism for the MLP layer.

<img src="https://github.com/wucong25/verl-data/blob/main/ascend_prefetch.png" alt="image-20251124173132677" style="zoom:50%;" />

### verl Framework Parameter Settings

Here are some memory-related settings and switches (note that the optimizations in this section may, to varying degrees, cause some degradation in throughput).

```bash
# Gradient Checkpointing
# Purpose: Saves device memory by recomputing activations, trading computation for memory. Intermediate activations are not saved during the forward pass and are recomputed during the backward pass, which significantly reduces device memory usage and allows for larger batch sizes.
actor_rollout_ref.model.enable_gradient_checkpointing=True \

# Parameter Offload
# Purpose: Offloads model parameters to CPU memory and loads them back to the GPU during training.
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \

# Optimizer Offload
# Purpose: Offloads optimizer states (such as Adam momentum) to the CPU. Optimizer states typically consume a large amount of device memory (for Adam, each parameter requires an additional 8 bytes); offloading saves device memory.
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \

# Free Cache Engine
# Purpose: Releases the inference engine's KV cache and weights during the training phase. This is the core optimization of the 3D-HybridEngine, allowing inference and training to alternate on the same GPU and significantly reducing device memory requirements.
actor_rollout_ref.rollout.free_cache_engine=True \

# Entropy Computation Optimization
# entropy_checkpointing: Enables recomputation for entropy computation during training to reduce peak device memory usage
# entropy_from_logits_with_chunking: Processes logits tensors in chunks (for example, groups of 2048 tokens) to avoid loading the entire [bsz*seq_len, vocab] tensor at once
actor_rollout_ref.actor.entropy_checkpointing=True \
actor_rollout_ref.ref.entropy_checkpointing=True \
actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \

# Inference Engine Device Memory Configuration
# gpu_memory_utilization: Controls the proportion of GPU device memory used by vLLM (0.90 = 90%)
# enforce_eager=False: Enables CUDA graphs to accelerate inference, but this uses additional device memory
actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
actor_rollout_ref.rollout.enforce_eager=False \
```

## NPU Tuning Reference Articles

Environment variables: [Environment Variable List - Ascend Extension for PyTorch 6.0.0 - Ascend Community](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/Envvariables/Envir_001.html)

Community performance tuning tutorial: [Performance Tuning Process - Ascend Extension for PyTorch 6.0.0 - Ascend Community](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0001.html)
