Ascend Quick Start Guide
=========================

**Last updated:** 2026/07/14.

Key Updates
--------------

- 2026/06/30: Added coverage of four common training and inference backend combinations, making it easier for users to quickly select the appropriate startup script during the quickstart phase.
- 2026/05/13: Separated the quick start and install guidance.
- 2025/12/11: Existing verl scenarios currently support automatic detection of NPU device types. When GPU scripts run on Ascend, explicitly setting the ``trainer.device=npu`` parameter is, in principle, no longer required. The new feature can still be used preferentially by setting ``trainer.device``, and the automatic detection capability is being gradually adapted.


Table of Contents
--------

- `Hardware Support <#hardware-support>`_
- `Qwen3-0.6B GSM8K GRPO Quick Start <#qwen3-06b-gsm8k-grpo-quick-start>`_
   - `Weight Preparation <#weight-preparation>`_
   - `Data Preparation <#data-preparation>`_
   - `How to Run <#how-to-run>`_
- `SGLang Backend Enablement Notes <#sglang-backend-enablement-notes>`_
   - `Converting vLLM Backend Scripts to SGLang <#converting-vllm-backend-scripts-to-sglang>`_

Hardware Support
------------------

- Atlas 200T A2 Box16
- Atlas 900 A2 PODc
- Atlas 800T A3



Qwen3-0.6B GSM8K GRPO Quick Start
---------------------------------

This guide provides a minimal GRPO training validation workflow based on GSM8K and Qwen3-0.6B for the Ascend NPU environment.

The documentation covers four commonly used training and inference backend combinations, making it easy for you to quickly select the appropriate startup script during the quickstart phase.

Before running the scripts in this document, ensure that you have completed the installation of the verl Ascend environment. For details on environment installation, refer to the `Ascend Installation Guide <./install_guidance.rst>`_.

A3 contains 2 die per card, while A2 contains 1 die per card. If you run the sample on an A3 machine, set ``n_gpus_per_node`` to 16.

All four scripts use the ``Qwen/Qwen3-0.6B`` model and the GSM8K dataset by default for basic pipeline validation.

Primarily used to check:

- Whether the verl entry point is available;
- Whether the data can be read;
- Whether the actor, rollout, and reference workers can be initialized;
- Whether the vLLM-Ascend/sglang rollout can generate;
- Whether the training pipeline can complete the first step.

Weight Preparation
~~~~~~~~~~~~~~~~~~~~

The weights must be downloaded from Hugging Face.

The default weight path read by the script is ``~/models/Qwen/Qwen3-0.6B``.

It is recommended to place the weights in this path, or modify the MODEL_PATH in the script to point to a local path.


Data Preparation
~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_dataset_path /download/path/hf_data/gsm8k/

Download the original GSM8K dataset from Hugging Face.

Generated files:

.. code-block:: text

   ~/data/gsm8k/train.parquet
   ~/data/gsm8k/test.parquet

Running Mode
~~~~~~~~~~~~~~

The relevant scripts are all placed in the ``tests/special_npu/quick_start/`` directory.

First, enter the verl directory: ``cd /your/path/verl``

Enable the CANN environment: If you customized the CANN path, modify the following enable command based on the custom path.

.. code-block:: bash

   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh

Quick Start currently provides four commonly used combinations of training and inference backends. You can select the corresponding script based on the training backend and rollout backend.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 60

* - Combination
  - Training backend
  - Rollout backend
  - Running method
* - vLLM + FSDP2
  - FSDP2
  - vLLM-Ascend
  - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_vllm_ascend.sh
* - vLLM + Megatron
  - Megatron
  - vLLM-Ascend
  - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_vllm_ascend.sh
* - SGLang + FSDP2
  - FSDP2
  - SGLang
  - bash tests/special_npu/quick_start/run_qwen3_0_6b_fsdp2_sglang_ascend.sh
* - SGLang + Megatron
  - Megatron
  - SGLang
  - bash tests/special_npu/quick_start/run_qwen3_0_6b_megatron_sglang_ascend.sh

For detailed parameter descriptions in the script, see `Training Configuration Parameters and Metrics <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/dev_guide/model_dev/parameter_and_metrics.md>`_

For multi-node task startup, see the `Multi-Machine Task Startup Practice Guide <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/model_support/examples/multi-machine_task_startup_practice.rst>`_.

SGLang Backend Enablement Guide
-------------------------------------------

Currently, verl parses common inference parameters. For details, see the ``ServerArgs`` initialization parameters in `async_sglang_server.py <../../../../verl/workers/rollout/sglang_rollout/async_sglang_server.py>`_.

Other `SGLang parameters <https://github.com/sgl-project/sglang/blob/v0.5.10/docs/advanced_features/server_arguments.md>`_ can be passed through ``engine_kwargs``.

vLLM Backend Script Conversion to SGLang
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to convert the vLLM backend inference script to SGLang yourself, add or modify the following parameters.

.. code-block:: bash

# Required
actor_rollout_ref.rollout.name=sglang \
+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \

# Optional
# Enable the inference EP. For detailed usage, see:
# https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README_CN.md
++actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
++actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="deepep" \

# Must be set to True when the MoE model uses multiple DP
+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False \

# chunked_prefill is disabled by default
+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1

