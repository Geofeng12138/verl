# NPU Advanced Features Guide

This document describes the advanced features and optimization capabilities of Ascend NPU in the verl ecosystem for developer reference.

Last updated: 05/13/2026.

---

## Contents

- [NPU Advanced Features Guide](#npu-advanced-features-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Advanced Features of Inference Backends](#1-advanced-features-of-inference-backends)
    - [1.1 vLLM Inference Backend](#11-vllm-inference-backend)
    - [1.2 SGLang Inference Backend](#12-sglang-inference-backend)
      - [Advanced Parameter Configuration](#advanced-parameter-configuration)
  - [2. Advanced Features of Training Backends](#2-advanced-features-of-training-backends)
    - [2.1 FSDP Training Backend](#21-fsdp-training-backend)
    - [2.2 Megatron Training Backend](#22-megatron-training-backend)
      - [MindSpeed Monkey Patch Framework Principles](#mindspeed-monkey-patch-framework-principles)
      - [Advanced Parameter Configuration for Megatron](#advanced-parameter-configuration-for-megatron)
        - [Memory and Computation Optimization](#memory-and-computation-optimization)
        - [Fused Operator Acceleration](#fused-operator-acceleration)
        - [Pipeline Parallelism Optimization](#pipeline-parallelism-optimization)
        - [Weight Management](#weight-management)
  - [3. Performance Optimization Features](#3-performance-optimization-features)
    - [3.1 Memory Optimization](#31-memory-optimization)
    - [3.2 Computation Acceleration](#32-computation-acceleration)
    - [3.3 Parallelism Strategies](#33-parallelism-strategies)
  - [4. Mixture of Experts (MoE) Features](#4-mixture-of-experts-moe-features)
    - [MoE Support in vLLM/SGLang Inference](#moe-support-in-vllmsglang-inference)
    - [MoE Support in Megatron Training](#moe-support-in-megatron-training)
  - [5. Limitations and Precautions](#5-limitations-and-precautions)
  - [Appendix: Parameter Quick Reference](#appendix-parameter-quick-reference)
    - [Inference Backend Parameter Quick Reference](#inference-backend-parameter-quick-reference)
    - [Training Backend Parameter Quick Reference](#training-backend-parameter-quick-reference)

---

## 1. Advanced Features of the Inference Backend

Currently, verl supports two mainstream inference backends, vLLM and SGLang, both of which can run on Ascend NPUs. The following lists the advanced feature parameters supported by each backend.

### 1.1 vLLM Inference Backend

Ascend supports the vLLM inference backend through the **vllm-ascend plugin**. This plugin follows the [RFC](https://github.com/vllm-project/vllm/issues/11162) and provides a pluggable interface to decouple Ascend NPU from vLLM.

---

### 1.2 SGLang Inference Backend

Ascend supports related features through continuous contribution and maintenance to the SGLang community, involving the following core components:

| Component | Description |
|:---|:---|
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of Ascend NPU-optimized inference kernels, including attention mechanisms, normalization, activation functions, LoRA adapters, and so on |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | The Ascend implementation of DeepEP, providing highly optimized expert parallelism (EP) communication kernels for MoE models |

#### Advanced Parameter Configuration

| SGLang parameter | Corresponding verl general parameter | Description |
|:---|:---|:---|
| `attention_backend` | `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **Attention backend selection** — Set it to `ascend` on NPU to use the Ascend-optimized kernels |
| `quantization` | `actor_rollout_ref.rollout.quantization` | **Quantization support** — Supports quantized model loading and inference |


For more SGLang NPU feature parameters, refer to the [SGLang community NPU feature support documentation](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features).

---

## 2. Advanced Features of the Training Backend

### 2.1 FSDP Training Backend

Ascend provides FSDP-related support capabilities through `torch_npu`.

### 2.2 Megatron Training Backend

Megatron is a model parallel training framework developed by NVIDIA. To run it on NPUs, you must install **MindSpeed** for underlying support. MindSpeed uses **Monkey Patch** technology to seamlessly replace key Megatron components, enabling NPU adaptation.

#### How the MindSpeed Monkey Patch Framework Works

**Entry point:**
```python
from mindspeed.megatron_adaptor import repatch
```

**Call chain:**
```
repatch
├── Execute the megatron_adaptor.py module import
├── Import the features_manager module
├── Execute mindspeed/features_manager/__init__.py
├── The @AutoExecuteFunction decorator is triggered
├── patch_features() executes automatically
└── Perform the apply_features_pre_patches and apply_features_patches operations
```

**Core components:**

| Component | Responsibility |
|:---|:---|
| `Patch` class | Implements dynamic replacement of functions/classes and supports stacking multiple decorators |
| `parse_path()` | Dynamically imports and creates modules |
| `MindSpeedPatchesManager` | Manages all patch registrations as a global singleton |
| `MindSpeedFeature` | The Feature base class, through which each feature integrates the patch system by inheritance |

#### Megatron Advanced Parameter Configuration

##### Memory and Computation Optimization

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | **Pipeline output deallocation** — Releases output data after tensors are sent to the next PP stage, reducing peak device memory usage. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | **Recompute granularity control** — Supports `full`, `selective`, or `none`. `full` recomputes the entire Transformer layer, `selective` recomputes only the core attention part, and the default value is `none`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | **Recompute method** — Requires `recompute_granularity=full`. Supports `uniform` or `block`, and the default value is `None`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | **Number of recompute layers** — Requires `recompute_granularity=full`. A larger value reduces device memory usage but increases computation cost. The value must be divisible by the number of model layers in the current process. |

##### Fusion Operator Acceleration

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | **Flash Attention** — Specifies whether to use Flash Attention to accelerate attention computation. The default value is `true`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | **Fused rotary position embedding** — Uses fused operators to accelerate RoPE computation. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | **Fused SwiGLU** — Uses fused operators to accelerate the SwiGLU activation function. The default value is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | **Persistent LayerNorm** — Uses a persistent strategy to optimize LayerNorm. The default value is `False`. |

##### Pipeline Parallelism Optimization

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | **Loss layer pipeline split** — Treats the loss layer as a standard Transformer layer in the split. The default is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | **Embedding layer pipeline split** — Treats the input embedding layer as a standard Transformer layer in the split. The default is `False`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | **Number of layers in the first stage** — Specifies the number of layers in the first pipeline stage. The default is `none`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | **Number of layers in the last stage** — Specifies the number of layers in the last pipeline stage. The default is `none`. |

##### Weight Management

| verl Parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.use_mbridge` | **MBridge weight conversion** — Enables mbridge for weight format conversion |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | **Distributed checkpoint** — Saves/loads weights in a distributed format, defaults to `False` |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | **Distributed weight path** — Path for loading distributed checkpoints, defaults to `null` |

---

## 3. Performance Optimization Features

### 3.1 Memory Optimization

| Feature | Inference/Training | Description |
|:---|:---|:---|
| KV Cache dynamic release (`free_cache_engine`) | Inference (vLLM) | Automatically unloads the KV Cache after the generation phase; enabled by default |
| Memory saver mode (`enable_memory_saver`) | Inference (SGLang) | Supports dynamic release and restoration of device memory; verl defaults to `True` |
| Parameter CPU offload (`param_offload`) | Training (FSDP/Megatron) | Offloads model weights to the CPU |
| Optimizer CPU offload (`optimizer_offload`) | Training (FSDP/Megatron) | Offloads optimizer states to the CPU |
| Chunked entropy computation (`entropy_from_logits_with_chunking`) | Training (FSDP) | Computes entropy values in chunks to reduce peak device memory usage |
| Entropy computation chunk size (`entropy_from_logits_chunk_size`) | Training (FSDP) | The chunk size for entropy computation |
| Entropy computation checkpointing (`entropy_checkpointing`) | Training (FSDP) | Enables checkpointing for entropy computation |
| Pipeline output release (`deallocate_pipeline_outputs`) | Training (Megatron) | Releases tensors that have been passed in pipeline parallelism scenarios |
| Activation recomputation (`recompute_granularity`) | Training (Megatron) | Supports three levels of granularity control: full, selective, and none |

### 3.2 Compute Acceleration

| Feature | Inference/Training | Description |
|:---|:---|:---|
| Chunked prefill (`enable_chunked_prefill`) | Inference (vLLM) | Large prefills are chunked and processed together with the decode batch |
| Prefix caching (`enable_prefix_caching`) | Inference (vLLM) | Automatically caches shared prefixes to reduce redundant computation |
| Flash Attention | Training (Megatron) | Uses Flash Attention to accelerate attention computation, enabled by default |
| Fused rotary position embedding (`use_fused_rotary_pos_emb`) | Training (Megatron) | Uses fused operators to accelerate RoPE |
| Fused SwiGLU (`use_fused_swiglu`) | Training (Megatron) | Uses fused operators to accelerate the SwiGLU activation function |
| Persistent LayerNorm (`persist_layer_norm`) | Training (Megatron) | Optimizes the LayerNorm execution strategy |
| Group GEMM (`moe_grouped_gemm`) | Training (Megatron) | Group GEMM optimization for MoE scenarios |

### 3.3 Parallelism Strategies

| Parallelism Type | vLLM | SGLang | FSDP | Megatron | Description |
|:---|:---|:---|:---|:---|:---|
| Data Parallelism (DP) | ✅ | ✅ | ✅ | ✅ | Parallelism across the data dimension |
| Tensor Parallelism (TP) | ✅ | ✅ | — | ✅ | Tensor splitting within a layer |
| Pipeline Parallelism (PP) | — | — | — | ✅ | Pipeline splitting across layers |
| Expert Parallelism (EP) | ✅ | ✅ | — | ✅ | Parallelism across the MoE expert dimension |
| Sequence Parallelism (SP/Ulysses) | ✅ | ✅ | ✅ | ✅ | Splitting across the sequence dimension to support long sequences |
| Context Parallelism (CP) | ✅ | — | — | ✅ | Parallel processing of context |

---

## 4. Mixture of Experts (MoE) Features

### vLLM/SGLang Inference MoE Support

- **Expert Parallelism (EP)** — Configured through the `ep_size` parameter, it distributes different experts across different NPU devices.
- SGLang provides highly optimized EP communication kernels through [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md).

### Megatron Training MoE Support

| verl parameter | Description |
|:---|:---|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallel (EP) size, default `1` |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | TP-extended EP size, default `null` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | **Group GEMM** — Uses Group GEMM to optimize expert computation in MoE scenarios, default `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | **Router data type** — Data type for the weighted average of the router and expert outputs, optional `fp32`/`fp64`, default `fp32`, improves stability in multi-expert scenarios |

---

## 5. Limitations and Precautions

1. **mbridge and VPP are mutually exclusive**
   - `actor_rollout_ref.actor.megatron.use_mbridge` and `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` (VPP) **do not support being enabled at the same time for now**
   - Because verl enables mbridge by default, you must manually set `use_mbridge` to `False` when using VPP

2. **FSDP1 vs FSDP2 Differences**
   - `forward_prefetch` and `use_orig_params` apply only to FSDP1
   - FSDP2 is the recommended default version. For API support, refer to the [Ascend PyTorch Release Notes](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md)

3. **Recompute parameter dependencies**
   - `recompute_method` takes effect only when `recompute_granularity='full'`
   - `recompute_num_layers` takes effect only when `recompute_granularity='full'`
   - When `recompute_method='uniform'`, `recompute_num_layers` indicates the number of Transformer layers in each recompute unit, which must be divisible by the number of model layers in the current process.

4. **SGLang NPU-specific configuration**
   - `attention_backend` must be set to `ascend` to use the Ascend-optimized kernels
   - `enable_memory_saver` is enabled by default in verl and does not require additional configuration

---

## Appendix: Parameter Quick Reference Table

### Inference Backend Parameters Quick Reference

| Parameter category | vLLM parameter | SGLang parameter | verl common parameter |
|:---|:---|:---|:---|
| Model path | `model_path` | `model_path` | `actor_rollout_ref.model.path` |
| Device memory control | `gpu_memory_utilization` | `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| Graph mode | `enforce_eager` | `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` |
| Quantization | `quantization` | `quantization` | `actor_rollout_ref.rollout.quantization` |
| Maximum sequence length | `max_model_len` | — | `actor_rollout_ref.rollout.max_model_len` |
| Maximum number of concurrent requests | `max_num_seqs` | `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` |
| Tokenizer | `skip_tokenizer_init` | `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` |
| Remote code | `trust_remote_code` | `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` |
| TP parallelism | `tp_size` | `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| DP parallelism | `dp_size` | `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` |
| EP parallelism | `ep_size` | `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` |

### Training Backend Parameters Quick Reference

| Parameter Category | FSDP Parameter | Megatron Parameter |
|:---|:---|:---|
| Parameter Offload | `fsdp_config.param_offload` | `megatron.param_offload` |
| Optimizer Offload | `fsdp_config.optimizer_offload` | `megatron.optimizer_offload` |
| Sequence Parallelism | `ulysses_sequence_parallel_size` | `context_parallel_size` |
| Flash Attention | — | `override_transformer_config.use_flash_attn` |
| Recompute Granularity | — | `override_transformer_config.recompute_granularity` |
| Distributed Checkpoint | — | `use_dist_checkpointing` |
