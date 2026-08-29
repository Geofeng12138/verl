Profiling Collection Guide
==================================================================================

Last updated: 07/13/2026.

This tutorial describes how to collect data on Ascend devices using the GRPO or DAPO algorithm with the FSDP or MindSpeed (Megatron) backend.

Configuration
--------------

Use two-level profile settings to control data collection.

- Global collection control: Use the configuration items in `verl/trainer/config/ppo_trainer.yaml` (FSDP) or `verl/trainer/config/ppo_megatron_trainer.yaml` (MindSpeed) to control the collection mode and steps.
- Role profile control: Control parameters such as collection through the configuration items in each role.

Global collection control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Control the number of collection steps and the mode through the parameters in `ppo_trainer.yaml`:

-  global_profiler: Controls the ranks and modes for profiling collection.

- tool: The collection tool used. Options include nsys, npu, torch, and torch_memory.

-  nsys: NVIDIA's official system-level performance analysis tool.
-  npu: The native performance analysis tool for Huawei Ascend chips.
-  torch: The built-in performance profiler in the PyTorch framework.
-  torch_memory: PyTorch's device memory trace analyzer (based on the device memory history snapshot feature).

-   steps: This parameter can be set to a list of collection steps, for example, [2, 4], which means collecting steps 2 and 4. If set to null, no collection is performed.
-   save_path: The path for saving the collected data. The default value is "outputs/profile".

Role profiler control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``profiler`` field of each role, you can control the collection mode for that role.

-  enable: Whether to enable performance profiling for this role.
-  all_ranks: Whether to collect data from all ranks.
-  ranks: The list of ranks from which to collect data. If it is empty, no data is collected.
-  tool_config: The configuration of the profiling tool used by this role.

Control the specific collection behavior through the parameters in ``profiler.tool_config.npu`` for each role:

-  level: Collection level. The options are level_none, level0, level1, and level2.

- `level_none`: Disables all level-based data collection (turns off `profiler_level`).
- `level0`: Collects high-level application data, low-level NPU data, and operator execution details on the NPU. After weighing data volume against analysis capability, `level0` is the recommended default configuration.
- `level1`: Adds CANN-layer AscendCL data and AI Core performance metrics on the NPU on top of `level0`.
- `level2`: Adds CANN-layer Runtime data and AI CPU metrics on top of `level1`.

- contents: A list of options that control what is collected, for example
   npu, cpu, memory, shapes, module, stack.

-   npu: Whether to collect device-side performance data.
-   cpu: Whether to collect host-side performance data.
-   memory: Whether to enable memory analysis.
-   shapes: Whether to record tensor shapes.
-   module: Whether to record framework-level Python call stack information. Compared with stack, module is recommended for recording call stack information because it produces lower performance overhead.
-   stack: Whether to record operator call stack information.

- analysis: Whether to enable automatic data parsing.
- discrete: Whether to use discrete mode.
- profile_token_start: This parameter takes effect only in the rollout role. It specifies the starting response token for collection during the rollout decoding phase. It takes effect when the parameter is valid (starting from 0, satisfying ``profile_token_end > profile_token_start``, and the range is within the response length).
- profile_token_end: This parameter takes effect only in the rollout role. It specifies the ending response token for collection during the rollout decoding phase (the right boundary is exclusive). It takes effect when the parameter is valid (starting from 0, satisfying ``profile_token_end > profile_token_start``, and the range is within the response length).

Example
-------

Disabling collection
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   global_profiler:
     steps: null # disable profile

End-to-End Profiling
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

global_profiler:
         steps: [1, 2, 5]
         save_path: ./outputs/profile
      actor_rollout_ref:
         actor:  # Configure profiler collection parameters for the actor role
            profiler:
               enable: True
               all_ranks: True
               tool_config:
                  npu:
                     discrete: True
                     contents: [npu, cpu]  # Controls the collection list; the default is cpu and npu. You can configure memory, shapes, module, and so on.

Training and inference phases are separated
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

global_profiler:
         steps: [1, 2, 5]
         save_path: ./outputs/profile
      actor_rollout_ref:
         actor:
            profiler:
               enable: True  # Set to True to collect data during the training phase
               all_ranks: False
               ranks: [0]  # Global Rank 0
               tool_config:
                  npu:
                     discrete: True
                     contents: [npu, cpu]
         rollout:
            profiler:
               enable: True  # Set to True to collect data during the inference phase
               all_ranks: False
               ranks: [0]  # Global GPU rank; it is mapped to the inference instance (replica) that owns this rank
               tool_config:
                  npu:
                     discrete: True  # Discrete mode must be enabled in Agent Loop mode
                     # Optional: Lightweight collection of inference data by response token range; if start/stop are not set, the entire rollout phase is collected
                     profile_token_start: 30
                     profile_token_end: 60
         # ref follows actor settings

Quick Start
--------------

Disable collection
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

            global_profiler.steps=null

End-to-End Profiling
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

global_profiler.tool=npu
        global_profiler.steps="[1, 2, 5]" # Steps to collect
        global_profiler.save_path=./outputs/profile
        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.all_ranks=False
        actor_rollout_ref.actor.profiler.ranks="[0]" # Collect rank 0 only
        actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True # Discrete mode is recommended; data for each stage is stored separately
        actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']" # Controls the collection list; the default is cpu and npu. You can configure memory, shapes, module, and so on.
        actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
        actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False # Disable automatic data analysis
        # rollout & ref follow actor settings


Lightweight collection of inference data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

global_profiler.tool=npu
      global_profiler.steps="[1, 2, 5]" # Steps to collect
      global_profiler.save_path=./outputs/profile
      actor_rollout_ref.actor.profiler.enable=True
      actor_rollout_ref.actor.profiler.all_ranks=False
      actor_rollout_ref.actor.profiler.ranks="[0]" # Collect rank 0 only
      actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True # The discrete mode is recommended. Data for each stage is stored separately.
      actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']" # Controls the collection list. The default values are cpu and npu. You can configure memory, shapes, module, and so on.
      actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
      actor_rollout_ref.actor.profiler.tool_config.npu.analysis=False # Disable automatic data analysis

actor_rollout_ref.rollout.profiler.enable=True
actor_rollout_ref.rollout.profiler.all_ranks=False
actor_rollout_ref.rollout.profiler.ranks="[0]" # Collect data from rank 0 only
# Optional: Lightweight collection of inference data, collected by response token range; if start/stop are not set, the entire rollout phase is collected
actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_start=30
actor_rollout_ref.rollout.profiler.tool_config.npu.profile_token_end=60
# ref follows actor settings

**Agent Loop Mode Description**:

In `Agent Loop <../../../../advance/agent_loop.rst>`_ mode, the performance data of the Rollout phase **must be collected in discrete mode**, in which case the Profiler is triggered by the inference engine backend.

1. Rank definition: The ranks in the rollout configuration are global GPU ranks (consistent with the training roles). Because each rollout instance (replica) spans ``world_size = tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size`` GPUs, each specified rank is mapped to the instance that owns it (``replica = rank // world_size``), and profiling is performed on the entire instance. For example, when ``tp=8``, ``ranks: [0, 8]`` profiles the instances that hold global ranks 0 and 8 (that is, replica 0 and replica 1).

2. Inference engine support: Currently, the vLLM and SGLang engines are supported without additional configuration. The details are as follows:

- vLLM engine: Automatically collects performance data from the AsyncLLM scheduling stack and inference processes. It does not support setting `analysis` (not parsed by default; requires offline parsing) or `profiler_level` (defaults to level1).
- SGLang engine: Automatically collects performance data from inference processes. It does not support the `memory` configuration item in `contents`. It does not support setting `analysis` (parsed by default) or `profiler_level` (defaults to level0).

**Fully Async Policy Mode Description**:

1. In `Fully Async Policy <https://verl.readthedocs.io/en/latest/advance/fully_async.html>`_ mode, `global_profiler.steps` represents the `step` after each `update_weights` round, which is consistent with synchronous mode, rather than the `mini-batch step` of a single round.

2. Because the AgentLoop collection capability is reused, the precautions in `Fully Async Policy <https://verl.readthedocs.io/en/latest/advance/fully_async.html>`_ mode are the same as those for AgentLoop.

Visualization
--------------

The collected data is stored in the `save_path` that you set, and you can visualize it using the `MindStudio Insight <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html>`_ tool.

Additionally, in Linux environments, the MindStudio Insight tool provides a `JupyterLab plugin <https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0130.html>`_ that offers a more intuitive and interactive interface. The advantages of the JupyterLab plugin are as follows:

- Seamless integration: MindStudio Insight can be run directly in a Jupyter environment, eliminating the need to switch platforms or copy server data, so you can use data as soon as it is collected.
- Quick startup: You can quickly start MindStudio Insight through the JupyterLab command line or graphical interface.
- Smooth running: Starting MindStudio Insight through the JupyterLab environment on Linux effectively resolves lag issues compared to whole-package communication, significantly improving the user experience.
- Remote access: MindStudio Insight supports remote startup, allowing you to connect to the service from a local browser for direct visual analysis, which alleviates the difficulties of uploading and downloading data for large model training or inference.

If the `analysis` parameter is set to `False`, offline analysis is required after collection.

.. code:: python

import torch_npu
    # Set profiler_path to the parent directory of the "localhost.localdomain_<PID>_<timestamp>_ascend_pt" directory
    torch_npu.profiler.profiler.analyse(profiler_path=profiler_path)


Advanced Guide: Fine-grained Profiling
----------------------------------------

Background and challenges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the configuration-file-based profiling method described above is convenient, it faces challenges in training scenarios with **long context** or **large global batch size**.
Within a complete training step, model computation exhibits high-frequency and repetitive characteristics:

1. Rollout phase: Generate Sequence is an autoregressive process that involves thousands of forward computations of the Decoder model.
2. Training phase: To control the peak device memory usage, verl typically uses a Micro-Batch strategy, which splits the large data stream into multiple micro-batches for computation.

- compute_log_prob (Actor/Ref): involves multiple rounds of pure forward propagation.
- update_policy (Actor/Critic): involves multiple rounds of forward and backward propagation.

This characteristic causes full profiling to generate a massive number of duplicate operator records, as shown in the following figure:

.. image:: https://raw.githubusercontent.com/mengchengTang/verl-data/master/verl_ascend_profiler.png
   :alt: Diagram showing that full profiling generates a large number of duplicate operator records

Even when using ``discrete`` mode, the performance data file for a single stage can still reach several terabytes, which causes **parsing failures** or **visualization tool lag**.

Solution: Critical Path Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To address the above issues, you can use the **critical path sampling** strategy: based on the API interfaces provided by `torch_npu.profiler <https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/atlasprofiling_16_0038.html>`_, directly modify the Python source code to collect only representative data segments (for example, a specific Decode Step or the first Micro-Batch).

**Important Notice**

1. This section involves directly modifying the source code. We recommend that you back up the files before making changes and restore them after debugging is complete.
2. When using code instrumentation for collection, be sure to **disable global collection** (``global_profiler: steps: null``) in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yaml`` to avoid Profiler conflicts.

1. Add a script to control the collection granularity
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

export PROFILE_STEP=2 # Collect profiling data for the specified step
export ROLLOUT_PROFILE=true
export UPDATE_PROFILE=true
export WITH_MODULES=false # Collect Python call stacks
export WITH_STACK=false # Collect operator call stacks
export WITH_MEMORY=false # Collect memory profiling data
export WITH_SHAPE=true # Collect tensor shapes
export PROFILE_RANKS=0 # Collect profiling data for rank 0
export UPDATE_PROFILE_PATH="./outputs/update_profile"
export ROLLOUT_PROFILE_PATH="./outputs/rollout_profile"

2. Fine-grained collection in the Rollout phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the vLLM or SGLang inference engine, you can control the ``schedule`` parameter to collect forward propagation performance data of the model at a specific token.

**vLLM engine**

- **Reference version**: vLLM v0.18.0, vLLM-Ascend v0.18.1
- **Modified file**: ``vllm-ascend/vllm_ascend/worker/worker.py``

.. code-block:: diff

      class NPUWorker(WorkerBase):

          def __init__(self, *args, **kwargs):
              # ... existing code ...

# profile collection
import os
import torch_npu
if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
    # Initialize profiler
    import torch_npu
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    self.profiler_npu = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        with_modules=os.environ.get('WITH_MODULES', "false") == "true",
        profile_memory=os.environ.get('WITH_MEMORY', "false") == "true",
        record_shapes=os.environ.get('WITH_SHAPE', "false") == "true",
        with_stack=os.environ.get('WITH_STACK', "false") == "true",
        experimental_config=experimental_config,
        # Skip the first 29 steps, warm up for 1 step, collect for 30 steps, and repeat once.
        schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('ROLLOUT_PROFILE_PATH'), analyse_flag=True)  # Path to save collected data, whether to analyze online
    )
    self.profiler_npu.start()

              # ... existing code ...

          def execute_model(self, scheduler_output=None, intermediate_tensors=None, **kwargs):
              # ... existing code ...
              output = self.model_runner.execute_model(scheduler_output,
                                                  intermediate_tensors)

import os
if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
    self.profiler_npu.step()  # Drive the schedule to collect data for some decode steps

              # ... existing code ...

**SGLang engine**

- **Reference version**: SGLang master branch
- **Modified file**: ``sglang/python/sglang/srt/model_executor/model_runner.py``

.. code-block:: diff

      # ... existing imports ...
  +   import torch_npu

      class ModelRunner:

          def __init__(self, *args, **kwargs):
              # ... existing init code ...

# profile collection
import os
import torch_npu
if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
    # Initialize profiler
    import torch_npu
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    self.profiler_npu = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        with_modules=os.environ.get('WITH_MODULES', "false") == "true",
        profile_memory=os.environ.get('WITH_MEMORY', "false") == "true",
        record_shapes=os.environ.get('WITH_SHAPE', "false") == "true",
        with_stack=os.environ.get('WITH_STACK', "false") == "true",
        experimental_config=experimental_config,
        # Skip the first 29 steps, warm up for 1 step, collect for 30 steps, and repeat once.
        schedule=torch_npu.profiler.schedule(wait=29, warmup=1, active=30, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('ROLLOUT_PROFILE_PATH'), analyse_flag=True)  # Path to save collected data, whether to analyze online
    )
    self.profiler_npu.start()

          def forward(self, forward_batch, **kwargs):
              # ... existing code ...

import os
if os.environ.get('ROLLOUT_PROFILE', "false") == "true":
    self.profiler_npu.step()  # Drive the schedule to collect data for some decode steps

              return output

3. Fine-grained collection for the update_policy (Actor & Critic) phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Update phase includes forward and backward propagation. Under the unified model engine, the mini-batch loop is driven by ``TrainingWorker.train_mini_batch`` in ``verl/workers/engine_workers.py``, which calls ``train_batch`` for each mini-batch.

**FSDP backend**

The FSDP backend supports setting the collection granularity for Mini-Batch and Micro-Batch.
For Mini-Batch level, instrument ``TrainingWorker.train_mini_batch``;
For Micro-Batch level, instrument the micro-batch loop in ``forward_backward_batch`` of the FSDP engine.

- **Modified file**: ``verl/workers/engine_workers.py``
  (``TrainingWorker.train_mini_batch``, at the Mini-Batch granularity) or
  ``verl/workers/engine/fsdp/transformer_impl.py``
  (``FSDPEngineWithLMHead.forward_backward_batch``, at the Micro-Batch granularity)

.. code-block:: diff

      class TrainingWorker(Worker, DistProfilerExtension):

          def __init__(self, config: TrainingWorkerConfig):
              # ...
  +           self.step = 1

          def train_mini_batch(self, data: TensorDict) -> TensorDict:
             # ...

import os
import torch_npu
if self.step == int(os.environ.get('PROFILE_STEP', 1)) and os.environ.get('UPDATE_PROFILE', "false") == "true":
    # Prepare the profiler
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    self.prof_npu = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        with_modules=os.environ.get('WITH_MODULES', "false") == "true",
        profile_memory=os.environ.get('WITH_MEMORY', "false") == "true",
        record_shapes=os.environ.get('WITH_SHAPE', "false") == "true",
        with_stack=os.environ.get('WITH_STACK', "false") == "true",
        experimental_config=experimental_config,
        # Collect only the first Mini Batch (including the computation of all Micro-Batches and one optimizer update)
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(os.environ.get('UPDATE_PROFILE_PATH'), analyse_flag=True)
    )
    if str(torch.distributed.get_rank()) in os.environ.get('PROFILE_RANKS', "0").split(','):
        self.prof_npu.start()

for batch_idx, mini_batch_td in enumerate(dataloader):
    # ... internally calls self.train_batch(mini_batch_td), which performs
    # Forward & Backward for each micro-batch inside the engine and completes
    # one optimizer update ...
    actor_output = self.train_batch(mini_batch_td)

+              if self.step == int(os.environ.get('PROFILE_STEP', 1)) and os.environ.get('UPDATE_PROFILE', "false") == "true":
+                  # Drive the schedule to collect data for the mini batch. To collect data for the micro batch, move self.prof_npu.step() into the micro batch loop.
+                  if str(torch.distributed.get_rank()) in os.environ.get('PROFILE_RANKS', "0").split(','):
+                      self.prof_npu.step()
+          # This mini batch ends.
+          self.step += 1


**Megatron backend**

The Megatron backend supports profiling at the Mini-Batch granularity. The entry point is also ``TrainingWorker.train_mini_batch``: the Megatron engine internally invokes Megatron's pipeline parallel forward/backward scheduling and executes one optimizer step.

- **File to modify**: ``verl/workers/engine_workers.py``
  (``TrainingWorker.train_mini_batch``) — identical to the FSDP code snippet above.
  It is recommended to rename the output directory (for example, ``./outputs/megatron_actor_update_profile``)
  to distinguish traces from different backends.

4. Fine-grained Profiling of the compute_log_prob (Actor & Ref) Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This stage calculates the probability distributions of the new and old policies. Under the unified model engine, the log-prob calculations for both actor and ref go through ``TrainingWorker.infer_batch`` and are ultimately dispatched to the ``BaseEngine.infer_batch`` of the corresponding backend engine.

**FSDP backend**

FSDP backend allows fine-grained control at the micro-batch level, enabling instrumentation within the micro-batch loop of the FSDP engine's forward process.

- **Modified file**: ``verl/workers/engine/fsdp/transformer_impl.py``
  (``FSDPEngineWithLMHead.forward_backward_batch`` / ``forward_step``)

.. code-block:: diff

# ... Import dependencies ...
+   import torch_npu

      class FSDPEngineWithLMHead(FSDPEngine):

          def forward_backward_batch(self, data: TensorDict, loss_function, forward_only=False):

+           role = "Ref" if forward_only and not self.optimizer_config else "Actor"
+           # Prepare the profiler (configuration as above, omitted)
+           experimental_config = torch_npu.profiler._ExperimentalConfig(...)
+           self.prof_npu = torch_npu.profiler.profile(
+               # ...  (configuration as above, omitted)
+               # wait=0, warmup=0, active=1: directly collect the first micro-batch
+               schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
+               on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./outputs/{role}_compute_log_prob", analyse_flag=True)
+           )

# forward_backward_batch is shared by ref and actor, distinguished by the role flag;
# To collect actor_compute_log_prob, change it to role == "Actor":
if role == "Ref":
    self.prof_npu.start()

              for micro_batch in micro_batches:

# ... original computation logic ...
                  with torch.no_grad():
                      output = self.forward_step(micro_batch, loss_function, forward_only=True)

+                   # Collect data for micro batches based on the driver schedule
+                   if role == "Ref":
+                       self.prof_npu.step()

                  # ...


**Megatron backend**

The Micro-Batch scheduling for the Megatron backend is managed internally by Megatron's pipeline parallelism ``forward_backward_func``. Fine-grained Micro-Batch-level profiling through simple code instrumentation is not currently supported. We recommend using the global profiler configuration for profiling.
