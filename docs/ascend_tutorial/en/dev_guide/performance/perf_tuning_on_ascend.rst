Ascend Performance Tuning Guide
====================================

Last updated:  01/29/2026.

Author:  `Xiaobo Hu <https://github.com/tardis-key>`_, `Haozhe Li <https://github.com/ZLiao097>`_

The performance tuning methods described in `Perf Tuning <https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst>`_ also apply to Ascend devices. This article focuses on tuning techniques specific to Ascend, including fused operator optimization, specific hardware configurations, and Ascend affinity features.

Fused Operators
--------------------------

Commonly Used Fused Operator List
**********************************

The optimization principle of fused operators is to merge the computation of multiple operators into a single operator through mathematically equivalent substitutions, reducing redundant computation and the number of dispatches, thereby improving performance. Several typical NPU fused operators are listed below, and they have all been applied to the Qwen2 and Qwen3 series models in `npu_patch.py`.

For the full fusion operators currently used in verl, see `npu_patch.py <https://github.com/verl-project/verl/blob/main/verl/models/transformers/npu_patch.py>`_.

Matrix Computation-Communication operator fusion (MC2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MC2 is a collective term for a series of computation-communication fusion operators in CANN. These operators fuse originally serial communication and computation operations, and optimize performance through internal splitting and pipeline parallel execution.

In vllm-ascend, you can specify the environment variable:

.. code-block:: sh

    export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1

In the forward computation, enable ``torch_npu.npu_mm_all_reduce_base`` in ``RowParallelLinear`` to merge the separate ``matmul`` and ``allreduce`` operations into a single fused operator.

`RotaryMul&RotaryMulGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0030.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface: ``torch_npu.npu_rotary_mul(x, r1, r2)``

Parameter description:

- x: q and k must be 4-dimensional inputs, typically in the shape ``[B, N, S, D]``, ``[B, S, N, D]``, or ``[S, B, N, D]``.

- r1: cos value. The shape must be 4-dimensional, typically ``[1, 1, S, D]``, ``[1, S, 1, D]``, or ``[S, 1, 1, D]``.

- r2: sin value. The shape of the input must be 4-dimensional, typically ``[1, 1, S, D]``, ``[1, S, 1, D]``, or ``[S, 1, 1, D]``.

`RmsNorm&RmsNormGrad <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0031.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface: ``torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor)`` 
Parameter description:

- self: Tensor type, with shape supporting 1 to 8 dimensions.

- gamma: Tensor type, usually the weight, with a shape that must match the trailing dimensions of self.

- epsilon: Float data type, used to prevent division by zero errors.

Output description:

- The first output is a Tensor, which is the final output y of the calculation formula.

- The second output is a Tensor, which is the intermediate result `rstd` of `rms_norm`, used for backward computation.

`Swiglu <https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0035.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

torch_npu interface: ``torch_npu.npu_swiglu(Tensor self, int dim=-1) -> (Tensor)``

Parameter description:

- self: Tensor type, with shape supporting 1-8 dimensions.

- dim: Int type, default is -1.

Output Description:

- Output as Tensor, the final output y of the calculation formula.

`GroupMatMul <https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function prototype:

.. code:: python

    npu_grouped_matmul(
        x, 
        weight, 
        *, 
        bias=None, 
        scale=None, 
        offset=None, 
        antiquant_scale=None, 
        antiquant_offset=None, 
        per_token_scale=None, 
        group_list=None, 
        activation_input=None, 
        activation_quant_scale=None, 
        activation_quant_offset=None, 
        split_item=0, group_type=None, 
        group_list_type=0, 
        act_type=0, 
        output_dtype=None, 
        tuning_config=None
    ) -> List[Tensor]

For detailed usage instructions, see the linked document in the title.

FSDP Backend Fused Operator Usage

In the ``verl/models/transformers/npu_patch.py`` directory, the available fused operators have already been replaced through patches, so they are used by default without requiring any additional operations.

Megatron Backend Fused Operator Usage
**********************************

Megatron's fused operators are integrated into MindSpeed and require specific parameters to be enabled:

1. **Flash Attention (must be enabled)**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
       ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True

2. **RotaryMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True

3. **RMSNorm**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True

4. **GroupMatMul**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

5. **Swiglu**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True

6. **Permute/Unpermute**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.fused_permute_unpermute=True

7. **MC2**
   ::

       +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=True

Ascend General Configuration
--------------------------

```
Operator Dispatch <https://www.hiascend.com/document/detail/zh/Pytorch/730/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md>`_
************************************************************************************************************************************************************************************************************
```

You can configure the optimization level of the task_queue operator submission queue through ``TASK_QUEUE_ENABLE``. The default is Level 1 optimization. This configuration reduces host submission time and can be used to mitigate the issue of excessive overall free time caused by submission.

.. image :: https://github.com/verl-project/verl-data/blob/main/images/ascend/perf_tuning_task_queue.png
    :width: 500px

Level 0: The downlink pipeline optimization is not enabled.

Level 1: Split the operator dispatch task into two stages. Place one part of the task (mainly the calls to aclnn operators) on a newly added secondary pipeline. The primary and secondary pipelines pass tasks through an operator queue and run in parallel with each other. This approach reduces the overall dispatch time through partial masking, improving end-to-end performance.

Level 2: Building on the Level 1 optimization, this level further balances the task load between the primary and secondary pipelines, mainly by migrating workspace-related tasks to the secondary pipeline, which provides better masking effects and greater performance gains. This configuration takes effect only in binary scenarios, and the recommended configuration value is Level 2 optimization.

`Communication Algorithm Orchestration Expansion <https://www.hiascend.com/document/detail/zh/canncommercial/850/maintenref/envvar/envref_07_0096.html>`_
************************************************************************************************************************************************************************************************************
Use the environment variable ``HCCL_OP_EXPANSION_MODE=AIV`` to configure the orchestration expansion location of communication algorithms. The following values are supported:

- **AI_CPU:** Indicates that the orchestration and expansion of the communication algorithm is located on the AI CPU on the Device side, where the Device side automatically selects the corresponding scheduler based on the hardware model.

- **AIV:** Indicates that the communication algorithm is orchestrated and expanded on the Vector Core on the Device side, and execution also occurs on the Vector Core.

- **HOST:** Indicates that the orchestration and expansion of the communication algorithm occurs on the Host-side CPU, while the Device side automatically selects the corresponding scheduler based on the hardware model.

- **HOST_TS:** Indicates that the communication algorithm is orchestrated and expanded on the Host-side CPU. The Host sends tasks to the Device's Task Scheduler, which schedules and executes them.

Inference Phase Tuning
--------------------------

Chunked Prefill in V1
***************************

VLLM V1 is enabled by default in the current VLLM version. Use the following configuration to enable Chunked Prefill:

.. code-block:: sh

    actor_rollout_ref.rollout.enable_chunked_prefill=True

For performance principles, refer to the `VLLM official documentation <https://docs.vllm.ai/en/v0.4.2/models/performance.html>`_.

Graph Mode
***************************

Similar to CUDA, NPU enables **ACL Graph** through the following configuration:

.. code-block:: sh

    actor_rollout_ref.rollout.enforce_eager=False

.. note::
    ACL Graph conflicts with the ``taskqueue Level 2`` mechanism, and **the two cannot be enabled at the same time**.

Training Phase Tuning
--------------------------

FSDP
**********************************

.. csv-table::
   :header: "FSDP", "Description"
   :widths: 30, 60

"/", "Only shard optimizer (Zero-1)"
SHARD_GRAD_OP, Shard gradients and optimizer (Zero-2)
"HYBRID_SHARD", "Shard weights, gradients, and optimizer (Zero-3)"
"2D device_mesh+HYBRID_SHARD", "Also known as HSDP (FSDP+DDP). For example, device_mesh=[2,8], where every 8 ranks form an FSDP group. FSDP sharding is performed within each group, and there are two groups in total. DDP is performed between the two groups, and gradients are synchronized through allreduce."
"2D device_mesh+HYBRID_SHARD_ZERO2", "The Zero2 version of HSDP"
NO_SHARD, DDP

FSDP does not support Zero-1. In VeRL, the device mesh value is determined based on the number of devices and ``actor_rollout_ref.actor.fsdp_config.fsdp_size``, with Zero-3 used for sharding by default. If the model is small (recommended for models smaller than 7B), you can set ``actor_rollout_ref.actor.fsdp_config.reshard_after_forward`` to ``True`` to use Zero-2 on FSDP/FSDP2 to optimize performance.

Megatron
**********************************

When the model is large, using Megatron as the training backend provides more flexible performance tuning.

When DP parallel device memory cannot accommodate the model, enable TP first to shard the model weights. If the model is still too large, then enable PP to shard it further. If the sequence is too long and causes activations to be too large, you can enable CP and SP for optimization. In MoE models, you can additionally enable EP to control expert sharding. If the experts are too small, to avoid sharding the weights too finely, you can enable ETP to avoid TP sharding for the MoE part, and instead distribute multiple complete experts across DP and TP.

TP, PP, EP, and ETP are used in the same way as in Megatron. To enable CP and SP on NPU:

- SP: ``Sequence Parallel`` further improves computational efficiency on top of Tensor Parallel. It is a parallel computing method that splits the sequence dimension of input data. On NPU, SP is invoked through MindSpeed:
  ::

      actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True

- CP: ``Context Parallel`` is a method for processing neural network activation values in parallel across multiple GPUs/NPUs. It partitions the input tensor along the sequence dimension. On NPUs, CP is invoked through MindSpeed (both parameters must be added together):
  ::

      actor_rollout_ref.actor.megatron.context_parallel_size
      actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size

Megatron-distributed optimizer
**********************************

When working with larger models, you typically need to shard the optimizer across each device within a DP domain to save device memory. To enable the distributed optimizer on NPU with the Megatron backend:

::

    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
