NPU FAQ
=========

Last updated: 05/13/2026.

This document summarizes common issues and solutions encountered when running VERL training and inference on NPUs.

Environment Configuration Issues
--------------------------------

### Q1: What should I do if the NPU device is not visible?

**Problem**：torch_npu.npu.is_available() returns False

**Solution**:

.. code-block:: bash

# Check device visibility
echo $ASCEND_RT_VISIBLE_DEVICES

# Set visible devices and disable Ray automatic device setting
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# Check the driver status
npu-smi info

Debugging and Diagnosis
--------------------------

### Q1: How do I enable NPU performance profiling?

Use VERL's built-in profiler:

.. code-block:: shell

   actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true \
   actor_rollout_ref.actor.profiler.tool_config.npu.contents=npu,cpu \
   actor_rollout_ref.actor.profiler.tool_config.npu.level=1 \
   actor_rollout_ref.actor.profiler.tool_config.npu.analysis=true

### Q2: How do I troubleshoot NPU training failures?

**Troubleshooting steps**:

1. Check the environment variable configuration
2. Verify device visibility
3. Check CANN version compatibility
4. View the specific error information in the logs
5. Reproduce the issue with a minimal example

**Enable detailed logging**:

.. code-block:: bash

# VERL framework logs
export VERL_LOGGING_LEVEL=DEBUG

# Ascend NPU logs (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# HCCL Communication Logs
export HCCL_DEBUG=INFO

Common Error Messages
------------

### Q1： "torch_npu detected, but NPU device is not available or visible"

**Cause**: The NPU driver is not installed correctly or the device is not visible.

**Solution**: Check the driver installation status and the ASCEND_RT_VISIBLE_DEVICES setting.

### Q2： "KeyError: decoder.layers.0.self_attention.q_layernorm.weight"

**Cause**: The MindSpeed version is too old.

**Solution**: Switch MindSpeed to 2.3.0_core_r0.12.1

### Q3： "AssertionError: Weight ... is too large to fit in the bucket"

**Problem Symptom**: During weight synchronization in distributed training, the following error occurs:

.. code-block:: text

   AssertionError: Weight model.embed_tokens.weight(torch.Size([151936, 4096]), torch.float32) is too large to fit in the bucket.
   Please increase rollout.update_weights_bucket_megabytes(2048 MB).

**Cause**: The size of a weight tensor in the model exceeds the default capacity of the weight transmission bucket (2048 MB). In the verl framework, model weights are chunked and packed into buckets (buffers) for transmission. When a single weight tensor exceeds the bucket size, the assertion check fails.

**Weight Size Calculation Method**:

Weight tensor memory usage (bytes) = product of the sizes of each dimension × number of bytes per element

The number of bytes corresponding to each data type is as follows:

- ``torch.float32`` → 4 bytes
- ``torch.float16`` / ``torch.bfloat16`` → 2 bytes
- ``torch.int8`` → 1 byte

For example, take ``model.embed_tokens.weight`` from this case:

.. code-block:: text

Tensor shape: torch.Size([151936, 4096])
Data type: torch.float32 (4 bytes)
Weight size = 151936 × 4096 × 4 = 2,483,027,968 bytes ≈ 2369 MB

Default bucket size = 2048 MB < 2369 MB → assertion failure triggered

**Solution**: When starting training, add the ``update_weights_bucket_megabytes`` parameter so that the bucket capacity exceeds the memory usage of the largest weight tensor:

.. code-block:: bash

   actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

**Parameter Value Selection Recommendations**:

1. **Memory usage of the largest weight tensor in the model**: Iterate through all model parameters, find the one with the largest ``nbytes``, and convert it to MB (divide by 1024²).

2. **Round up to a power of 2**: For easier memory allocation and management, it is recommended to round the calculation result up to the nearest power of 2 (for example, 2048, 4096, 8192, and so on). For example, if the maximum weight is 2369 MB, use 4096 MB.

3. **Reserve an appropriate margin**: Considering memory alignment and runtime overhead, we recommend that the bucket size be at least 1.2 to 1.5 times the size of the largest weight, and then round up to a power of 2.

4. **Mind the memory limit**: The bucket size directly affects the memory usage of worker nodes, and setting it too large can cause OOM. Choose the smallest value possible while still meeting the weight transfer requirements.

**Recommended Values for Common Models**:

.. list-table::
   :header-rows: 1

* - Model size
  - Typical maximum weight shape
  - Recommended bucket size
* - 7B (such as Qwen2)
  - [151936, 4096] float32
  - 4096 MB
* - 14B
  - [152064, 5120] float32
  - 4096 MB
* - 72B
  - [152064, 8192] float32
  - 8192 MB

### Q4: Checkpoint loading fails on non-shared storage; common.pt / .metadata / metadata.json cannot be found

**Issue**: When using the verl + Megatron backend in a multi-machine environment with **non-shared storage**, saving checkpoints works normally, but reloading fails with an error indicating that the following files cannot be found:

.. code-block:: text

   FileNotFoundError: common.pt
   FileNotFoundError: .metadata
   FileNotFoundError: metadata.json

**Cause**: The current checkpoint mechanism does not fully support non-shared storage. This is reflected in the following ways:

- **Distributed training weights are saved per node**, and each node saves only the weight shards it is responsible for, rather than saving all weights only on the main node.
- However, metadata files such as ``common.pt``, ``.metadata``, and ``metadata.json`` are **saved only on the node that performs the save operation** (usually the node where rank 0 resides), and other nodes do not have these files locally.
- When loading a checkpoint, each node needs to read these metadata files to restore the model state, but under non-shared storage, these files do not exist in the local paths of other nodes, causing the loading to fail.

**Temporary workaround**: Manually copy the metadata file from the saving node to all other nodes:

.. code-block:: bash

# Assume the checkpoint is saved in the /path/to/ckpt/ directory on the rank 0 node
# Copy the metadata file from the rank 0 node to all other nodes

# Files to copy
/path/to/ckpt/common.pt
/path/to/ckpt/.metadata
/path/to/ckpt/metadata.json

# Example: Use scp to copy to another node
scp /path/to/ckpt/common.pt node1:/path/to/ckpt/
scp /path/to/ckpt/.metadata node1:/path/to/ckpt/
scp /path/to/ckpt/metadata.json node1:/path/to/ckpt/

# Repeat the preceding steps for all nodes

**Precautions**:

- After each checkpoint save, you must copy the metadata files again, because the save operation can update the contents of these files.
- If checkpoints are saved frequently during training (for example, automatically saved by step), it is recommended that you write a script to automatically trigger the copy after saving, to avoid omissions.
- For a long-term solution, wait for framework-level support for loading checkpoints from non-shared storage, so that metadata files can be automatically synchronized to all nodes.

Reference materials
--------

- `Ascend Performance Tuning Guide <../dev_guide/performance/perf_tuning_on_ascend.rst>`_
- `Ascend Quick Start Guide <../get_start/quick_start.rst>`_
- `NPU-CI Integration Guide <../contribution_guide/ascend_ci_guide.rst>`_
- Ascend NPU documentation: https://www.hiascend.com/document
- CANN toolkit documentation: https://www.hiascend.com/software/cann

Get more help
--------------

If the preceding FAQ does not resolve your issue, do the following:

1. View the complete error log
2. Search for similar issues in GitHub Issues
3. Provide detailed error information and environment configuration
4. Provide a minimal reproducible example