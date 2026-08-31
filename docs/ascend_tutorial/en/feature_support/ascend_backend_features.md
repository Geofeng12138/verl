# Ascend Backend Features Guide
==================================================================================

Last updated: 03/03/2026.

Ascend provides comprehensive support for the verl ecosystem. This document describes the adaptation work for verl on NPUs and the backend feature support, for your reference.

---

## Inference Backend

Currently, verl supports two mainstream inference backends, vLLM and SGLang, both of which can run on Ascend NPUs.

### 1. vllm:

Ascend supports the vLLM inference backend through the vllm-ascend plugin, which is the recommended method for the vLLM community to support the Ascend backend. It follows the [[RFC]](https://github.com/vllm-project/vllm/issues/11162) and provides a pluggable interface that decouples Ascend NPU from vLLM.

#### Parameter Feature Support

| vLLM Parameter | Corresponding verl General Parameter | Description |
| --- | --- | --- |
| `model_path` | `actor_rollout_ref.model.path` | Path to the model weight file. |
| `gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` | Controls the amount of GPU memory that can be used in each stage. It is specified as a fraction between 0.0 and 1.0, where: - 0.8 means 80% of the total GPU memory - 1.0 means 100% of the total GPU memory (not recommended, as no buffer is reserved) |
| `enforce_eager` | `actor_rollout_ref.rollout.enforce_eager` | Disables graph mode. The default value in verl is False. |
| `enable_chunked_prefill` | `actor_rollout_ref.rollout.enable_chunked_prefill` | Chunked prefill allows large prefills to be split into smaller chunks and batched together with decode requests. |
| `free_cache_engine` | `actor_rollout_ref.rollout.free_cache_engine` | Unloads the KVCache after the generation phase is deployed. The default value is True. |
| `max_model_len` | `actor_rollout_ref.rollout.max_model_len` | The maximum sequence length that the model can handle. It limits the maximum length of a single input sequence. |
| `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size` | Tensor parallelism size. |
| `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` | Data parallelism size. |
| `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` | Expert parallelism size. |
| `node_rank` | None. It is automatically calculated based on the actual instances and the number of devices. | Node rank within an instance. |
| `load_format` | `actor_rollout_ref.rollout.load_format` | The format of the model weights to load. |
| `disable_log_stats` | `actor_rollout_ref.rollout.disable_log_stats` | Controls whether rollout statistics logs are recorded. |
| `nnodes` | None. It is automatically calculated based on the actual instances and the number of devices. | The number of nodes contained in each instance. |
| `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` | Whether to allow custom models defined on the Hub and written in your own modeling files. |
| `max_num_seqs` | `actor_rollout_ref.rollout.max_num_seqs` | The maximum number of running requests. |
| `max_num_batched_tokens` | `actor_rollout_ref.rollout.max_num_batched_tokens` | The maximum total number of tokens that can be processed in a single batch. |
| `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` | Skips the tokenizer initialization and passes `input_ids` directly to the inference request. |
| `enable_prefix_caching` | `actor_rollout_ref.rollout.enable_prefix_caching` | Enables automatic prefix caching. |
| `quantization` | `actor_rollout_ref.rollout.quantization`. The default value is None. | Quantization method. |

### 2. sglang:

For the SGLang inference backend, Ascend supports the relevant features through continuous development and maintenance directly within the SGLang community.
In addition, using SGLang in verl involves the following components:

| Component | Description |
| --- | --- |
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of Ascend NPU SGL-optimized inference kernels, including attention mechanisms, normalization, activation functions, LoRA adapters, and so on. |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | An Ascend implementation of DeepEP, providing highly optimized expert parallel (EP) communication kernels for MoE models. |

#### Parameter Feature Support

In verl, inference backend parameters are enabled through the rollout config, which includes common parameters and custom arguments passed through `engine_kwargs`.

The following lists the SGLang feature parameters commonly configured in verl. For more parameter details, refer to [SGLang community NPU feature support](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features).

| sglang parameter | Corresponding verl general parameter | Description |
| --- | --- | --- |
| model_path | actor_rollout_ref.model.path | Path to the model weight file |
| mem_fraction_static | actor_rollout_ref.rollout.gpu_memory_utilization | Proportion of memory used for static allocation (model weights and key-value cache memory pool) |
| disable_cuda_graph | actor_rollout_ref.rollout.enforce_eager | Disables graph mode; the default value in verl is False |
| enable_memory_saver | None; the default value in verl is True | Allows the use of release_memory_occupation and resume_memory_occupation to save memory |
| base_gpu_id | None; automatically calculated based on the actual instance and number of devices | Initial ID used to allocate compute device resources on each instance |
| gpu_id_step | None; the default value is 1 | Difference between consecutive compute device IDs used |
| tp_size | actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size | Tensor parallelism degree |
| dp_size | actor_rollout_ref.rollout.data_parallel_size | Data parallelism degree |
| ep_size | actor_rollout_ref.rollout.expert_parallel_size | Expert parallelism degree |
| node_rank | None; automatically calculated based on the actual instance and number of devices | Node ranking in the instance |
| load_format | actor_rollout_ref.rollout.load_format | Format of the model weights to load |
| dist_init_addr | None; automatically calculated | Host address used to initialize the distributed backend |
| nnodes | None; automatically calculated based on the actual instance and number of devices | Number of nodes contained in each instance |
| trust_remote_code | actor_rollout_ref.model.trust_remote_code | Whether to allow custom models defined on the Hub and written into your own modeling files |
| max_running_requests | actor_rollout_ref.rollout.max_num_seqs | Maximum number of running requests |
| log_level | None; the default value is error | Log level of the logger |
| skip_tokenizer_init | actor_rollout_ref.rollout.skip_tokenizer_init | Skips tokenizer initialization and passes input_ids to inference requests |
| skip_server_warmup | None; the default value is True | Skips warmup |
| quantization | actor_rollout_ref.rollout.quantization; the default value is None | Quantization method |
| attention_backend | actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend | Attention kernel; set it to ascend on NPU |

---

## Training Backend

### 1. FSDP

Ascend provides FSDP-related support capabilities through torch_npu. For the current PyTorch API support level, refer to the [release notes](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md).

#### FSDP1
##### Parameter Feature Support
| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | Whether to offload model weights to the CPU. The default value is `False`. |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | Whether to offload the optimizer state to the CPU. The default value is `False`. |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | Controls the parameter behavior after the forward pass, balancing memory and communication. The default value is `True`: parameters are resharded after the forward pass and fully gathered again during the backward pass. |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | The number of NPUs in each FSDP sharding group. The default value `-1` indicates automatic. |

| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`  |Prefetches the all-gather for the next forward pass before the current forward computation completes. This is only used with FSDP1. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | Indicates whether FSDP uses the module's original parameters for initialization. This is only used with FSDP1. The default value is False.|
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`|The Ulysses sequence parallelism size.|
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`|Computes entropy in chunks to reduce peak device memory usage. The default value is False.|
| `actor_rollout_ref.actor.entropy_from_logits_chunk_size`|The chunk size for entropy computation. The default value is 2048.|
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`|Enables recomputation for entropy computation during training to reduce peak device memory usage. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.forward_only` |Indicates whether to perform forward computation only. The default value is False.|

#### FSDP2
##### Parameter Feature Support
| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | Whether to offload model weights to the CPU. The default value is False. |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | Whether to offload the optimizer state to the CPU. The default value is False. |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | Controls the parameter behavior after the forward pass, balancing memory and communication. The default value is True: parameters are resharded after the forward pass and fully gathered again during the backward pass. |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | The number of NPUs in each FSDP shard group. The default value of -1 indicates automatic. |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | The Ulysses sequence parallel size. |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | Computes entropy through chunking to reduce peak device memory usage. The default value is False. |
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing` | Enables recomputation for entropy computation during training to reduce peak device memory usage. The default value is False. |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | Whether to perform only the forward pass. The default value is False. |



### 2. Megatron

Megatron is a training framework repository introduced by NVIDIA that focuses on model parallelism. If a repository (for example, Verl) uses Megatron as its training backend and you also want to run that repository on NPU, you need to install MindSpeed additionally to provide underlying support. The following sections describe how MindSpeed seamlessly replaces key components in Megatron to adapt it to NPU.

The underlying replacement mechanism of MindSpeed uses Monkey Patch technology.

* MindSpeed Monkey Patch framework

In verl, the patch is triggered by `from mindspeed.megatron_adaptor import repatch`. The call stack is as follows:

```
from mindspeed.megatron_adaptor import repatch
├── Execute the megatron_adaptor.py module import
├── Import the features_manager module
├── Execute mindspeed/features_manager/__init__.py
├── @AutoExecuteFunction decorator triggers
├── patch_features() executes automatically
└── Perform `apply_features_pre_patches` and `apply_features_patches` operations
```

The `Patch` class is the core of the entire patch system, implementing dynamic replacement of functions and classes.

~~~python
class Patch:
~~~

The `parse_path` method implements dynamic module import and creation.

~~~python
def parse_path(module_path, function_name, create_dummy):
~~~

The patch system supports the stacking of multiple layers of decorators.

~~~python
def apply_patch(self):  
    final_patch_func = self.orig_func  
    if self.patch_func is not None:  
        final_patch_func = self.patch_func  

# Apply all decorators  
for wrapper in self.wrappers:  
    final_patch_func = wrapper(final_patch_func)

* MindSpeedPatchesManager class

`MindSpeedPatchesManager` manages all patches as a global singleton.

~~~python
class MindSpeedPatchesManager:  
    patches_info: Dict[str, Patch] = {}
~~~

* Feature integration mode

Each feature integrates into the patch system by inheriting the `MindSpeedFeature` base class.

~~~python
class MindSpeedFeature:
    """Base class for mindspeed features."""

    def __init__(self, feature_name: str, optimization_level: int = 2):
        self.feature_name = feature_name.lower().strip().replace('-', '_')
        self.optimization_level = optimization_level
        self.default_patches = self.optimization_level == 0

    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return (self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, None)) \
            or self.default_patches

    def register_args(self, parser: ArgumentParser):
        """Register cli arguments to enable the feature."""
        pass

    def pre_validate_args(self, args: Namespace):
        """Validate the arguments of mindspeed before megatron args validation
        and store some arguments of the mindspeed temporarily,
        in case that megatron validate fails.
        for example:
            ```python
            origin_context_parallel_size = args.context_parallel_size
            args.context_parallel_size = 1
            ```
        """
        pass

    def validate_args(self, args: Namespace):
        """Restore the arguments of the mindspeed.

        for example:
        ```python
        args.context_parallel_size = origin_context_parallel_size
        ```
        """
        pass

    def post_validate_args(self, args: Namespace):
        """validate mindspeed arguments after megatron arguments validation."""
        pass

    def pre_register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        """Register all patch functions before import megatron"""
        pass

    def register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        """Register all patch functions the feature is related."""
        pass

    def incompatible_check(self, global_args, check_args):
        """Register all incompatible functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and getattr(global_args, check_args, None):
            raise AssertionError('{} and {} are incompatible.'.format(self.feature_name, check_args))

    def dependency_check(self, global_args, check_args):
        """Register all dependency functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and not getattr(global_args, check_args, None):
            raise AssertionError('{} requires {}.'.format(self.feature_name, check_args))

    @staticmethod
    def add_parser_argument_choices_value(parser, argument_name, new_choice):
        """Add a new choice value to the existing choices of a parser argument."""
        for action in parser._actions:
            exist_arg = isinstance(action, argparse.Action) and argument_name in action.option_strings
            if exist_arg and action.choices is not None and new_choice not in action.choices:
                action.choices.append(new_choice)
~~~

#### Parameter Feature Support
| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | Whether to offload the model optimizer to the CPU. The default value is False. |
| `actor_rollout_ref.actor.megatron.use_mbridge` | Whether to enable mbridge: when set to True (the default), the engine constructs a `bridge` and passes it to the checkpoint manager, which enables reading and writing to `model/huggingface/`. When `save_contents` / `load_contents` includes `hf_model`, the manager requires `bridge` to be non-empty (which is usually the case when this parameter is True). This parameter can be enabled together with `use_dist_checkpointing`, writing both the HF tree and the `model/dist_ckpt/` shards in the same checkpoint. When set to False, `hf_model` is generally not present; if only the `model` slot uses `dist_checkpointing`, you must also set `use_dist_checkpointing=True`. |
| `actor_rollout_ref.actor.megatron.param_offload` | Whether to offload model weights to the CPU. The default value is False. |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | The tensor parallel size. The default value is 1. |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | The pipeline parallel size. The default value is 1. |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | The expert parallel size. The default value is 1. |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | The TP-extended EP size. The default value is null. |
| `actor_rollout_ref.actor.context_parallel_size` | The sequence parallel size. The default value is 1. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | Whether to deallocate the output data after the tensor is sent to the next pipeline stage, which reduces peak device memory usage. The default value is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | Whether to use persistent LayerNorm. The default value is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | Whether to use Group GEMM. The default value is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | The data type used for routing and for the weighted average of expert outputs. Using fp32 or fp64 can improve stability, especially when the number of experts is large. The default value is fp32. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | If set to True, the loss layer is treated as a standard Transformer layer in the pipeline parallel partitioning and placement strategy. The default value is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | If set to True, the input embedding layer is treated as a standard Transformer layer in the pipeline parallel partitioning and placement strategy. The default value is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | The granularity for recomputing activations. The options are `'full'`, `'selective'`, and `'none'`. `full` recomputes the entire Transformer layer, while `selective` recomputes only the core attention part of the Transformer layer. The default value is `'none'`. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | This parameter takes effect only when `recompute_granularity` is set to `'full'`. The options are `'uniform'` and `'block'`. The default value is None. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | This parameter takes effect only when `recompute_granularity` is set to `'full'`. The default value is None. If `recompute_method` is set to `uniform`, this parameter specifies the number of Transformer layers in each uniformly divided recomputation unit. For example, you can specify `--recompute_granularity full --recompute_method uniform --recompute_num_layers 4`. A larger `recompute_num_layers` reduces device memory usage but increases computational cost. Note: the number of model layers in the current process must be divisible by `recompute_num_layers`. The default value is None. |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | When set to True, the `model` slot uses Megatron `dist_checkpointing` shards (`model/dist_ckpt/`). This parameter is independent of `use_mbridge`: both can be set to True to save/load shards and export to HF simultaneously. The default value is False. |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | The distributed weight path. The default value is null. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | Whether to use Flash Attention. The default value is true. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | Whether to use fused rotary position embedding. The default value is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | Whether to use fused SwiGLU. The default value is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | The number of layers in the first pipeline stage. The default value is none. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | The number of layers in the last pipeline stage. The default value is none. |

Note: `actor_rollout_ref.actor.megatron.use_mbridge` and `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` (VPP) are not supported simultaneously. Because verl enables mbridge by default, when using the VPP parameter, manually set `actor_rollout_ref.actor.megatron.use_mbridge` to False.

### 3. VeOmni

VeOmni is a unified reinforcement learning training backend designed for efficient training of large-scale models. It is built on FSDP and provides a rich set of parallelism strategies and optimization features, making it particularly suitable for MoE models and large-scale distributed training scenarios.

#### Parameter Feature Support

| verl parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.veomni.param_offload` | Whether to offload model weights to the CPU. The default value is `False`. |
| `actor_rollout_ref.actor.veomni.optimizer_offload` | Whether to offload optimizer states to the CPU. The default value is `False`. |
| `actor_rollout_ref.actor.veomni.fsdp_size` | The number of NPUs in each FSDP shard group. The default value `-1` means automatic. |
| `actor_rollout_ref.actor.veomni.ulysses_parallel_size` | The Ulysses sequence parallelism size. The default value is `1`. |
| `actor_rollout_ref.actor.veomni.expert_parallel_size` | The expert parallelism size. The default value is `1`. |
| `actor_rollout_ref.actor.veomni.mixed_precision` | Whether to enable mixed precision training. The default value is `true`. |
| `actor_rollout_ref.actor.veomni.enable_full_shard` | Whether to enable full sharding (ZeRO-3). The default value is `true`. |
| `actor_rollout_ref.actor.veomni.forward_prefetch` | Whether to prefetch the all-gather for the next forward pass before the current forward computation completes. The default value is `true`. |
| `actor_rollout_ref.actor.veomni.attn_implementation` | The attention implementation. Supported values include `eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `veomni_flash_attention_2_with_sp`, `veomni_flash_attention_3_with_sp`, and `native-sparse`. |
| `actor_rollout_ref.actor.veomni.moe_implementation` | The MoE implementation. Supported values are `eager` or `fused`. The default value is `fused`. |
| `actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation` | The cross-entropy loss implementation. The default value is `eager`. |
| `actor_rollout_ref.actor.veomni.rms_norm_implementation` | The RMSNorm implementation. The default value is `eager`. |
| `actor_rollout_ref.actor.veomni.swiglu_mlp_implementation` | The SwiGLU MLP implementation. The default value is `eager`. |
| `actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation` | The rotary position embedding implementation. The default value is `eager`. |
| `actor_rollout_ref.actor.veomni.load_balancing_loss_implementation` | The MoE load balancing loss implementation. The default value is `eager`. |
| `actor_rollout_ref.actor.veomni.use_torch_compile` | Whether to use torch compile. The default value is `false`. |
| `actor_rollout_ref.actor.veomni.forward_only` | Whether to perform forward computation only. The default value is `false`. |
| `actor_rollout_ref.actor.veomni.enable_fsdp_offload` | Whether to enable CPU offloading for FSDP. The default value is `false`. |
| `actor_rollout_ref.actor.veomni.enable_reentrant` | Whether to use reentrant gradient checkpointing. The default value is `false`. |
| `actor_rollout_ref.actor.veomni.ckpt_manager` | The checkpoint manager. The default value is `dcp`. |
| `actor_rollout_ref.actor.veomni.init_device` | The device used for model weight initialization. Supported values include `cpu`, `cuda`, `meta`, and `npu`. The default value is `meta`. |
| `actor_rollout_ref.actor.veomni.activation_gpu_limit` | The activation memory limit (in GB) allowed on the GPU during activation offloading. The default value is `0.0`. |
| `actor_rollout_ref.rollout.moe_load_balance_metrics_interval` | The interval for reporting MoE expert load metrics on the rollout side. The default value is `0` (disabled). You must also enable `actor_rollout_ref.rollout.enable_rollout_routing_replay` to record routing decisions. |

#### Router Replay Support

VeOmni backend supports the Router Replay feature for MoE models, configured through `actor_rollout_ref.actor.veomni.router_replay`:

| Parameter | Description |
| --- | --- |
| `mode` | Router replay mode, which supports `disabled` (disabled), `R2` (records and replays routing decisions), and `R3` (records and replays at the rollout side) |
| `record_file` | The file path for recording routing decisions, required in R2/R3 modes |
| `replay_file` | The file path for loading routing decisions to replay, required in replay mode |

#### Usage Example

The VeOmni backend is particularly well-suited for GRPO training of large-scale MoE models. A typical configuration is as follows:

```bash
# Set the VeOmni training backend
model_engine=veomni

# Configure the parallel strategy
actor_rollout_ref.actor.veomni.fsdp_size=16
actor_rollout_ref.actor.veomni.ulysses_parallel_size=1
actor_rollout_ref.actor.veomni.expert_parallel_size=1

# Configure memory optimization
actor_rollout_ref.actor.veomni.param_offload=True
actor_rollout_ref.actor.veomni.optimizer_offload=True

# Configure operator implementations
actor_rollout_ref.actor.veomni.attn_implementation=veomni_flash_attention_2_with_sp
actor_rollout_ref.actor.veomni.moe_implementation=fused
```

#### Key Features

- **Efficient parallel strategies**: Supports flexible combinations of data parallelism, Ulysses sequence parallelism, and expert parallelism
- **Memory optimization**: Supports parameter offloading, optimizer offloading, and activation offloading to effectively reduce device memory usage
- **MoE optimization**: Provides fused MoE implementations and Router Replay functionality to improve MoE model training efficiency
- **Operator optimization**: Supports multiple attention and MLP operator implementations, allowing you to select the optimal implementation based on the hardware
- **Flexible deployment**: Supports NVIDIA GPU and Huawei Ascend NPU, providing good cross-platform compatibility
