# Precision Alignment Guide

When performing reinforcement learning (RL) training in the VeRL framework, **precision alignment** is a critical step to ensure that the training process is reproducible and debuggable.

This document summarizes the methods for aligning precision between NPU and GPU on VeRL for reference.

Last updated: 05/09/2026.

## 1. Environment and Weight Alignment

### 1.1 Dependency Version Alignment

VeRL and transformers versions must be strictly aligned; otherwise, the precision results are directly affected.

For other critical dependencies (torch, megatron, vllm), if strong alignment is not possible, prioritize keeping them consistent or similar.

### 1.2 Model Weight Alignment

Check whether the model weights and the config.json file are completely consistent.


## 2. Input Data Alignment

Add the following configuration to the verl training startup script:

```bash
data.shuffle=False
data.validation_shuffle=False
```


## 3. Configuration Alignment

When aligning precision between NPU and GPU, check whether the configurations are fully aligned. This includes:
1. Directly comparing the configurations written in the scripts
2. Saving logs during the run, collecting the configurations printed to the screen for comparison, and comparing whether the default parameter configurations are consistent. Ensure that the key parameters are aligned.


## 4. Fixed Determinism

### 4.1 Fixed Random Seed

Install `msprobe` in the environment:

```bash
pip install mindstudio-probe
```

Add a deterministic function at the beginning of the worker file:

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

### 4.2 Fixed Communication Environment Variables

In multi-device communication scenarios:

- In HCCL communication mode (default scenario):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

Under LCCL communication (enabled by setting `export HCCL_OP_EXPANSION_MODE="AIV"`):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

For single-device scenarios without communication:

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0



## 5. Verify Training Accuracy

### 5.1 Training Instrumentation

**Pinning** retains the input and output data of the current stage, making it easier to compare and analyze the results. When troubleshooting precision issues, pinning is required to assist in locating the problem. A common pinning method is to directly dump the data from the rollout stage.

**Step 1: Generate baseline data in a GPU environment**

First run the GPU script once, with the following configuration enabled:

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```
You can save the inference results of each step as a JSONL file.

**Step 2: Reproduce and verify in the NPU environment**

Enable the following parameters on the NPU, reuse the sequence generated in the previous step, and run end-to-end:

```bash
skip.rollout.enable=True \
skip.rollout.dump_dir=/path/to/rollout_dump \
```

**Step 3: Compare Metrics**

With identical stubbed inputs, consistent training configurations, and fixed randomness, compare whether the rewards/pg_loss/grad_norm values differ between NPU and GPU.


## 6. Verify inference precision

### 6.1 resharding

Before inference officially begins, vLLM performs a **dummy run**, which evaluates the device memory usage during inference by inferring one token, and then allocates device memory accordingly. You can specify the `load_format` parameter during vLLM LLM initialization to determine whether the weights for the dummy run are randomly initialized (`dummy`) or real weights (`safetensors`). In VeRL, you specify this parameter through **actor_rollout_ref.rollout.load_format**.

When garbled inference output occurs, if the engine initialization method is **load_format=dummy**, there is a high probability that sharding is problematic. Even if switching to safetensors produces normal output, the sharding issue still exists and requires a forward pass comparison.


### 6.2  Inference Result Alignment

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

Save the inference result of each step to a JSONL file. You can directly open the JSONL file to quickly confirm whether the inference result of the entire network is garbled, which is used to isolate inference precision issues.


Before dumping inference data, if reproducing the inference accuracy issue consumes significant resources, you can first try to reduce the cost of reproducing the issue by shrinking the reproduction scale, thereby decreasing the amount of data that needs to be dumped and compared. In scenarios with multiple batches and long sequences, you can attempt to reproduce the issue by sending single-batch requests and reducing the sequence length.


## 7. Dump Comparison

After using the [precision debugger](../../../en/dev_guide/precision_analysis/precision_debugger.md) to locate the stage where the problem occurs, you can use the msprobe tool to dump data for detailed troubleshooting.

During inference or training, the model may encounter numerical instability issues, such as outputs deviating from expectations, abnormal generation, or even NaN/Inf values. To identify the root cause, you need to finely monitor the model execution path, collect intermediate features, weights, activations, and inputs and outputs of key layers, and record context information such as prompts, tensor dtypes, and hardware configurations. By capturing these core tensors and metadata, you can systematically trace the source of precision degradation or numerical errors.




