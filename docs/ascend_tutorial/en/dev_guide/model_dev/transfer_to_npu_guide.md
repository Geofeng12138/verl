# Model Migration to NPU Guide

Last updated: 05/14/2026

This article provides developers with complete practical experience in migrating models from GPU to NPU or independently adapting models on NPU, covering the entire process from preliminary preparation, component integration, precision alignment, performance optimization, to long-running evaluation.

## 1. Prerequisites

Set up a basic runtime environment that supports NPU operation, ensuring that models load normally and datasets can be read smoothly. This serves as the foundation for subsequent migration debugging and business validation.


### 1.1 Software and Hardware Environment and Dependency Configuration

Refer to the official document [Ascend Installation Guide](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/get_start/install_guidance.rst); if the versions of the inference engines vllm, vllm_ascend, and the training engines Megatron, MindSpeed, and transformers that the model depends on differ from those in the tutorial, **use the versions that the model is actually adapted to**.

### 1.2 Model Weights

BF16 is the **default mixed-precision training data type** for training backends such as FSDP and Megatron in the VeRL framework. The Ascend NPU environment uniformly uses **BF16** as the baseline precision format, and weights must be dequantized to BF16 for alignment. Currently, A2 and A3 models **do not support FP8 precision training** and support only BF16 precision. Ascend 950 series products will enable FP8 low-precision training capabilities in a future release.

### 1.3 Data Preparation

The data must be preprocessed into parquet format by following [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html): (1) ensure that it contains the necessary fields for computing reinforcement learning rewards; (2) enable faster reading.

## 2. Integration of All Components

The VeRL framework adopts a decoupled architecture design that separates the inference engine, training engine, and weight synchronization bridge (Checkpoint Engine). This design enables deep separation of computation and data, providing a flexible extension foundation for migrating and adapting models to Ascend NPUs. When migrating and adapting models to NPUs, we recommend first completing the independent adaptation and validation of the inference engine, training engine, and Megatron-Bridge components. After each component runs stably, proceed with the integration and debugging of the full VeRL pipeline. For specific feature support of different VeRL inference and training backends on Ascend NPUs, refer to the [Ascend Backend Features Guide](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/feature_support/ascend_backend_features.md).

### 2.1 Inference Engine Adaptation

The VeRL inference engine uses a layered architecture design. Through abstract interfaces and the factory pattern, it provides flexible support for multiple mainstream inference backends, such as vLLM and SGLang. When adapting the inference engine for migration from GPU to NPU, follow the recommended workflow below:

Before running the full VeRL training pipeline on NPU, we recommend that you first review the official model deployment tutorials for [vllm-ascend](https://github.com/vllm-project/vllm-ascend/tree/main/docs/source/tutorials/models) and [sglang](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/basic_usage), and prioritize getting the **single-instance inference pipeline** working. This allows you to fully verify the **basic inference functions**, including model loading and initialization, Tokenizer loading, single-turn and batch generation, stop-word termination, and long-context inference. After you confirm that the underlying inference engine is stable and available, you can then integrate it with the VeRL training workflow.

### 2.2 Training Engine Selection and Adaptation

VeRL mainline code abstracts the training engine into the `Engine` class, decoupling scheduling logic from the underlying training implementation through a standardized interface layer. This architecture supports flexible, plug-and-play integration of multiple training backends such as FSDP and Megatron, without modifying VeRL core algorithms and scheduling logic, significantly reducing migration and adaptation costs.

The NPU is automatically detected through the `is_npu_available` interface, and the corresponding NPU device adaptation patch is applied automatically. Currently, you only need to set `model_engine=fsdp/megatron` to switch the training backend to FSDP or Megatron with one click. The system automatically loads the NPU adaptation logic for the corresponding backend, and no additional code changes are required. In VeRL, Ascend has adapted and optimized Megatron. For specific feature configurations, refer to the [verl-MindSpeed feature documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md).

### 2.3 Megatron-Bridge Adaptation

The Megatron-Bridge is mainly used to convert HuggingFace weights, which the inference engine depends on, and the mcore weights required by Megatron-Core bidirectionally within the VeRL framework. You can enable this feature with the following configuration:

```
actor_rollout_ref.actor.megatron.use_mbridge=True
actor_rollout_ref.actor.megatron.vanilla_mbridge=False
```

Megatron-Bridge has natively adapted a large number of mainstream model architectures in the community. For the supported model list, refer to the [supported model](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/models/README.md). When you perform model migration and adaptation in an Ascend NPU environment, you can complete the basic configuration based on the existing community capabilities. However, some models with special architectures and scenarios still require additional customized adaptation.

Using the DSA (DeepSeek Sparse Attention) sparse attention structure as an example, this section describes the customization and adaptation method. Ascend MindSpeed supports DSA capabilities based on the absorption matrix. This feature requires splitting the original `linear_kv_up_proj` operator in Megatron into two independent operators: `linear_k_up_proj` and `linear_v_up_proj`. The weights required for the split must be generated by converting from `self_attn.kv_b_proj.weight` in HuggingFace format. However, the native PR mentioned above does not adapt to this operator splitting logic.

Therefore, you need to manually adapt the relevant weight conversion logic to ensure that the absorption matrix can be loaded and take effect normally. Only when the absorption matrix is available can you properly enable the [sparse\_flash\_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/sparse_flash_attention) and [lightning\_indexer](https://gitcode.com/cann/ops-transformer/tree/master/attention/lightning_indexer) fusion operators. By introducing these two fusion operators, you can significantly reduce memory access frequency, optimize memory usage, and improve computational performance. This ultimately enhances the operating efficiency of large model training and inference pipelines while reducing resource overhead.

### 2.4 End-to-End Network Functionality

Complete the inference engine adaptation verification and training engine adaptation development. Refer to the [Training Configuration Parameters and Metrics Description](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/dev_guide/model_dev/parameter_and_metrics.md) to configure the relevant parameters for the inference engine and training engine based on actual business requirements. Complete the end-to-end VeRL functionality integration and ensure stable operation of the entire process.

## 3. Precision Alignment

The precision issue diagnosis chain for large model reinforcement learning is complex and influenced by many factors. Various precision issues are typically introduced by **training phase, inference phase, and training-inference consistency** problems. **Precision alignment** is the core key to ensuring reproducible training workflows and debuggable issues.

For training and inference phase precision alignment, refer to the official documentation: [Precision Alignment Guide](../precision_analysis/precision_alignment.md). Therefore, this section does not repeat the basic phase alignment process. Instead, it **focuses on training-inference consistency scenarios**, using the msprobe precision tool to carry out precision alignment practices and problem diagnosis.

### 3.1 Precision Monitoring Configuration

After the full network runs successfully, enable the precision monitoring parameters by setting `actor_rollout_ref.rollout.calculate_log_probs=True`. During training, focus on the following key metrics to evaluate training-inference consistency and model training stability:

* **Training-inference consistency reference metrics**:
  * `training/rollout_probs_diff_mean` (mean rollout probability difference): under normal model convergence, this metric should remain within 0.01. If the value consistently exceeds 0.01 or deviates significantly from the GPU baseline, a training-inference accuracy anomaly can be identified.
  * `training/rollout_probs_diff_max` (maximum rollout probability difference)
  * `training/rollout_actor_probs_pearson_corr` (Pearson correlation coefficient between rollout and actor probabilities)
* **Model training stability metrics**:
  * `actor/grad_norm`: monitor whether it shows an overall downward trend to determine whether model training is converging normally.

Additionally, the configuration parameter `trainer.rollout_data_dir=./rollout_dump/` is used to save the intermediate Rollout results during training. By manually inspecting the exported Rollout data, you can verify whether the model responses meet expectations and check for garbled output or repeated answers, which further confirms that the inference engine is correctly adapted.

### 3.2 Collecting Accuracy Data

When the training/rollout_probs_diff_mean exceeds the reasonable threshold of 0.01, or shows a significant deviation from the GPU baseline benchmark, further root cause analysis requires collecting data using the [msprobe](../../../en/dev_guide/precision_analysis/precision_debugger.md) precision tool.

### 3.3 Troubleshooting and Alignment Practices for Training-Inference Differences

After data collection is complete, first read the `construction.json` file to perform module-level data comparison. Ensure that the input data of `layer.0.input_layernorm` is completely consistent first, and then verify module by module and layer by layer to locate the first position where training and inference outputs become inconsistent.

For large models, tiny numerical differences accumulate and amplify layer by layer, leading to significant differences in training and rollout results. In extreme cases, the same token can have output probabilities of 0 and 1 during training and rollout, respectively. Therefore, align every difference point as closely as possible to achieve exact equality.

After locating the differing nodes, adapting the modification plan is also a key challenge. Because open-source communities in the industry provide multiple different implementations of the relevant modules, to ensure the correctness of the model implementation logic, you need to consult authoritative source code and technical reports from multiple sources and determine the final alignment plan comprehensively.

#### 3.3.1 Common Training/Inference Inconsistencies

In large model reinforcement learning practice, the typical root causes of training-inference inconsistency can be summarized into the following five categories:

1. **Framework implementation inconsistencies**: These are caused by differences in the implementation logic between the training and inference frameworks. Sometimes they are "semantically correct" (for example, different operator splitting methods that are mathematically equivalent), and sometimes they are "semantically incorrect" (for example, a missing scaling factor or an extra operation). You must strictly identify these by examining the source code and technical reports.
2. **Precision type differences**: For example, the training side uses BF16 throughout, while the inference side implicitly upcasts to FP32 for computation in sensitive operators such as normalization and then downcasts, which leads to truncation errors.
3. **Hyperparameter inconsistencies**: For example, the hardcoded `eps` value in the LayerNorm module is not unified.
4. **Parallel strategies**: Tensor parallelism during training versus continuous batching during inference leads to differences in the order of floating-point accumulation.
5. **Randomness control**: Implementation deviations of Dropout and sampling strategies between the training and inference stages.

The following lists typical cases of training-inference inconsistency encountered during the migration and adaptation of the GLM-5 model.

#### 3.3.2 Case 1: Inconsistent Framework Implementation of the FFN Activation Function

Compare from top to bottom, and locate the first inconsistency at the MLP activation function output in the first layer.

Inference already uses the NPU-optimized `npu_swiglu` fused operator, but training still executes the native GLU small operator implementation.

* **Root cause**: Although the `swiglu` enablement configuration was added to the VeRL parameters, the Megatron-Bridge did not explicitly configure `provider.bias_activation_fusion=True` in the NPU adaptation PR, which prevented the code from entering the NPU fused operator branch.
* **Fix**: Add a configuration item in Megatron-Bridge so that the training side correctly invokes the fused operator:
  ```
  +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
  ```

#### 3.3.3 Case 2: Inconsistency Between indexer_k_norm Precision and Hyperparameters

During the strict alignment process, a precision type mismatch with the hyperparameters was found at `indexer_k_norm`.

* **Precision difference**: On the inference side, LayerNorm includes an upcast to fp32 operation `F.layer_norm( x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)`, while the training side uses the Megatron implementation in BF16. The small differences accumulate across layers and cannot be ignored.
* **Fix**: Unify the training-side code by adding upcast and downcast operations.
* **Hyperparameter difference**: On the GLM5 inference side, vLLM inherits the DeepSeek-V3.2 logic, where the EPS value for `k_norm` is hardcoded to `1e-6`; however, the training engine and the official technical report consistently use `1e-5`.
* **Fix**: Change the inference-side EPS to `1e-5` to align with the training side.

```
self.k_norm=LayerNorm(self.head_dim,eps=1e-6 -> 1e-5)
```

#### 3.3.4 Case 3: Missing and Redundant Logic in the lightning_indexer Module

The investigation found that the `lightning_indexer` module has inconsistencies between its training and inference implementations, specifically:

* **Missing (inference-side omission)**: The inference side lacks the scaling logic for `weights`. Reference the standard implementations in Megatron training, slime, and transformers, all of which include this scaling; therefore, add it on the inference side to align with the forward pass:

```
weights, _ = self.weights_proj(x)
+weights = weights * (self.n_head**-0.5) * (self.head_dim**-0.5)
```

* **Redundancy (unnecessary error implementation on the training side)**: The Megatron implementation on the training side includes an extra `rotate_activation` (Hadamard transform). After consulting extensive resources, it is confirmed that this operation is specific to quantization scenarios and is an incorrect implementation in BF16 format. Refer to [Transformer PR#45017](https://github.com/huggingface/transformers/pull/45017) to remove this redundant logic on the training side.

```
class DSAIndexer(MegatronModule):
    def forward_with_scores(
-		q = rotate_activation(q)
-		k = rotate_activation(k)
```

### 3.4 General Routing Stability Solution for MoE Large Models

In a typical RL training workflow, an efficient inference engine such as vLLM is used for sample sampling, and the sampled data is then fed into a training framework such as Megatron for model training optimization.

For conventional dense models, implementation and environment differences between inference and training frameworks cause only minor numerical deviations; however, this issue is dramatically amplified in **large-scale MoE models**. The root cause lies in the MoE dynamic routing mechanism: minor differences in framework implementation or runtime environment can cause the same input token to be assigned to completely different expert combinations, leading to vastly divergent computation paths.

This inconsistency in routing decisions can lead to unstable RL training for MoE models. It makes the "experience" obtained during the inference phase completely different from that of the training phase, distorting the optimization signal and ultimately leading to catastrophic consequences.

To address this common problem, the industry has introduced the **Routing Replay** mechanism. Its core idea is to lock the expert routing path at a specific stage, shielding routing decisions from minor perturbations, thereby ensuring training stability. Currently, there are two mainstream variants: R2 and R3.

* **（1）Vanilla Routing Replay (R2)**: (corresponding to `actor_rollout_ref.actor.megatron.router_replay.mode="R2"`, and for VeOmni, `actor_rollout_ref.actor.veomni.router_replay.mode="R2"`)

* **Mechanism**: During the gradient update phase, the training engine replays the expert paths computed in the previous sampling phase.
* **Function**: This primarily mitigates the impact of **policy staleness** on routing. As the policy updates, the routing computed by the current forward pass may differ from the routing used when generating the old data. R2 maintains the coherence of the optimization signal by replaying the old routing.
* **（2）Rollout Routing Replay (R3)**: (corresponding to `actor_rollout_ref.actor.megatron.router_replay.mode="R3"`)

* **Mechanism**: Captures the routing distribution of the inference engine during sequence generation and replays it directly into the training engine.
* **Purpose**: Simultaneously addresses both **training-inference bias** and **policy staleness**. It ensures that the expert paths used to compute the loss during training are absolutely consistent with the expert paths used during actual inference generation.

Therefore, whether it is R2, which focuses on mitigating outdated strategies, or R3, which achieves full-chain alignment, the Routing Replay mechanism effectively bridges the routing gap between inference and training frameworks. In aligning training and inference consistency for **large-scale MoE models**, this mechanism has become a core means of ensuring precision alignment and training stability. Currently, mainstream large models such as DeepSeek-V3.2, GLM-5, and MiMo-V2 have all adopted Routing Replay technology in R3 mode.

Therefore, for large-scale MoE models, the R3 mode, which provides more thorough alignment, is generally recommended in actual configurations:

```
actor_rollout_ref.actor.megatron.router_replay.mode="R3" \
actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
```

VeOmni backend: switch to `actor_rollout_ref.actor.veomni.router_replay.mode="R3"`. The top-level `actor.router_replay` has been removed and no longer takes effect.

## 4. Performance Optimization

When optimizing large language model RL (reinforcement learning) training performance on Ascend NPUs, you can prioritize the official documentation for basic configuration tuning: [perf_tuning.rst](https://github.com/verl-project/verl/blob/04833f01/docs/perf/perf_tuning.rst). For more efficient optimization, follow the standardized workflow of **data collection → bottleneck identification → configuration tuning → iterative validation**. This workflow significantly improves throughput in core stages such as Rollout, Reward, and Update, while effectively reducing resource idle time and load imbalance. For specific performance analysis and tuning operations, strictly follow the official guidelines below:

1. [Ascend Performance Analysis Guide](../performance/ascend_performance_analysis_guide.md)
2. [Profiling Collection Guide](../performance/ascend_profiling.rst)

### 4.1 Inference Performance Optimization

The Rollout phase is the core inference stage of large model RL training, and its inference time accounts for the majority of the entire training process. The following are common performance optimization methods for this phase:

1. **Enable graph mode**: Graph mode compiles and optimizes the entire computation graph in advance, enabling deep optimizations such as operator fusion, memory reuse, and constant folding, which significantly improves execution efficiency.
2. **Accelerate operator dispatch through CPU core binding**: CPU core binding improves operator dispatch efficiency. Since vllm-ascend v0.18.0rc1, this capability is enabled by default on ARM-based Ascend servers.
3. **Configure the HCCL communication algorithm to AIV mode**: Set the environment variable `HCCL_OP_EXPANSION_MODE` to `AIV` mode to specify that the orchestration and expansion logic of the communication algorithm runs on the Device-side Vector Core compute unit.
4. **Enable asynchronous scheduling**: This eliminates the gap between two consecutive `execute_model` executions by the Worker, allowing the Worker to directly obtain the scheduled `SchedulerOutput` for model inference without blocking and waiting for scheduling.

The corresponding configuration parameters are as follows:

# Enable graph mode
actor_rollout_ref.rollout.enforce_eager=False
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
# Bind CPU cores
++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
# Enable asynchronous scheduling
++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
```

### 4.2 Training Performance Optimization

The Update phase of large model RL training is characterized by significant variation in sequence length and high device memory consumption. In addition to basic operator fusion, you need to combine sequence parallelism with memory-computation trade-off strategies to break through bottlenecks. For common training performance optimization features, refer to the [MindSpeed-verl documentation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/user-guide/verl.md) to enable them. Core optimization methods include:

1. **Operator fusion**: Enable fused operators such as RoPE, SwiGLU, RMSNorm, and DSA. Operator fusion reduces computation overhead and device memory usage, improving training efficiency.
2. **Remove padding**: In RL training, Response lengths vary across samples. Traditional padding strategies lead to a large amount of ineffective computation. Enabling Remove padding packs multiple short sequences to fill the tensor, greatly improving the utilization of NPU compute units (MFU).

## 5. Evaluation and Verification

After training is complete, you must evaluate and validate the target dataset to ensure that the business results after model migration meet the requirements. The evaluation steps are the same for different models. The following uses GLM-5 as an example to describe the evaluation process in detail (using the AISBenchmark tool, which supports evaluation of multiple inference backends such as vllm and sglang).

The evaluation uses the mathematics dataset aime2025 and the graduate-level professional science dataset gpqa to verify that scores improve in the target direction and that no catastrophic knowledge forgetting occurs in unrelated directions.

### 5.1 Installing aisbench

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
```

### 5.2 Downloading the Evaluation Dataset

```shell
# On the Linux server, in the tool root directory
cd path/to/benchmark/ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```

### 5.3 Modify the AISBench Configuration Code to Enable vLLM/SGLang Inference Evaluation

Open the `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py` file. This is the inference evaluation configuration file. It is recommended that the output length `max_out_len` remain consistent with the training `max_response_len`.

```shell
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="/path/to/GLM-5", # Modify to the GLM-5 model path
        model="GLM-5",
	    stream=True,
        request_rate = 0,
	    use_timestamp=False,
        max_seq_len=2048,
        retry = 2,
	    api_key="",
        host_ip = "localhost", # IP address of the inference service
        host_port = 12890 , # Port of the inference service
        max_out_len = 8192,  # Maximum output token length
        batch_size=48, # Maximum concurrency for inference
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0,
            seed = 1234,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
```

### 5.4 Starting the Inference Server on Multiple Machines

Refer to the [vllm_ascend GLM5 guide](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md#multi-node-deployment) to start a two-node A3 inference service. Keep `host_port` consistent with the configuration in the previous section, and set `max_model_len` to the sum of `max_prompt_length` and `max_response` used during training.

### 5.5 Starting the vLLM Evaluation Task

Run the following command to start the online inference evaluation task, which calls the deployed vLLM inference backend and loads the corresponding model configuration to complete the automated evaluation:

```
ais_bench --models vllm_api_stream_chat --datasets aime2025_gen_0_shot_chat_prompt
```

After training, the model achieves steady improvements in core capability metrics: evaluation scores on the AIME2025 mathematical reasoning dataset rise steadily, and the model also delivers continuous score gains on the GPQA graduate-level professional science dataset, with no knowledge degradation or catastrophic forgetting. The training optimization results meet expectations.

| Evaluation Dataset | GLM5-base | 10step | 15step | 40step | 50step |
| ---------- | --------- | ------ | ------ | ------ | ------ |
| aime2025   | 47.5      | 49.17  | 49.17  | 48.33  | 52.5   |
| gpqa       | 64.65     | 68.81  | 68.43  | 69.07  | 71.21  |

## 6. Summary

This article comprehensively covers the full workflow of migrating large models from GPU to Ascend NPU or independently adapting them on NPU. It is mainly divided into five key stages: environment setup, component integration, accuracy alignment, performance optimization, and evaluation validation. It provides developers with actionable and reusable operation guides and problem-solving solutions.

During the preparation phase, focus on controlling the environment dependency versions, model weight precision, and dataset format to lay the foundation for subsequent adaptation. In the component integration phase, follow the principle of validating single components first and then integrating the full network. Prioritize the stable adaptation of inference, training engines, and weight conversion tools. For special model structures, complete customized modifications. Precision alignment is the core of migration adaptation. Monitor the training-inference consistency metrics closely. Resolve common differences such as framework implementation and precision types through module-by-module troubleshooting. For MoE models, enable the Routing Replay mechanism to ensure training stability. Follow a standardized process for performance optimization, focusing on the core stages of inference and training. Improve efficiency and reduce resource consumption through graph mode, operator fusion, and other techniques. Finally, validate through standardized evaluation to ensure that the business effects meet the requirements after model migration, with no knowledge degradation.

Overall, following the workflow in this guide can effectively reduce the cost of NPU migration and adaptation, help you avoid common pitfalls, and achieve stable, efficient operation of large models on Ascend NPUs.
