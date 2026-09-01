# Ascend Performance Analysis Guide

Last updated: 02/24/2026.

## Background

With the release of DeepSeek-R1, large model reinforcement learning (RL) training has attracted widespread attention. In the Ascend NPU environment, the verl framework has accumulated extensive performance tuning experience. This guide systematically summarizes methodologies, including performance data collection and analysis, to help developers use the MindStudio toolchain more efficiently and achieve performance optimization in reinforcement learning scenarios.

### Overview of the Reinforcement Learning Computation Flow

1. **Rollout**: The policy (actor) model generates responses (response sequences) based on the input prompt sequences through inference.
2. **ref logprob**: Based on the prompt and the generated response, the reference model computes the ref logprob for KL divergence calculation.
3. **logprob**: Based on the prompt and the generated response, the actor model computes the logprob for importance sampling.
4. **reward**: Based on the prompt and the generated response, the reward model evaluates the reward value R_N.
5. **update**: Based on the computed R_N, ref logprob, and logprob, the optimization function and policy gradient are calculated to update the actor model.

![rl_data_stream](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/rl_data_stream.png)

## Enabling the profiling tool

### Enabling Method

For enabling and configuration tutorials, refer to the [Profiling Collection Guide](./ascend_profiling.rst).

## Performance Analysis Methodology

### Overall Performance Overview Analysis

#### 1. Analysis of Long-Duration Tasks and Resource Bubbles

- **Operation**: Use MindStudio Insight to load profiling data, automatically identify different computation stages, and locate long-duration tasks and NPU resource bubbles through the pipeline diagram on the RL tab.
- **Value**: Quickly understand the time consumption proportion of different stages.
- **Effect demonstration**:

![Bubble_analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Bubble_analysis.png)

#### 2. Load Balancing Analysis

- **Operation**: Use MindStudio Insight to directly view MSTX profiling data and observe the load balancing across different DP ranks during the Rollout phase.
- **Value**: Quickly identify load imbalance issues.
- **Result demonstration**:

![Load_Balancing_Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Load_Balancing_Analysis.gif)

#### 3. Cluster-Level Performance Analysis

- **Operation**: Use the rl_analysis feature of MSTT to generate a cluster Timeline thumbnail and observe the overall time consumption of each stage.
- **Value**: Gain a macro-level understanding of cluster performance bottlenecks.
- **Operation guide**: [rl_analysis usage documentation](https://gitcode.com/Ascend/mstt/raw/pre-research/profiler/msprof_analyze/docs/features/rl_analysis.md)
- **Effect demonstration**:

![Cluster%20Performance%20Analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Cluster%20Performance%20Analysis.png)

### Fine-grained Analysis

#### Performance Analysis

- **Operation**: You can load Profiling data through the MindStudio Insight Windows or Linux version.
- **Value**: MindStudio Insight supports analyzing task scheduling efficiency, operator execution performance, computing resource utilization, collective communication performance, and so on. Its Timeline view provides task decomposition and Overlap analysis capabilities (**a core feature unique to MindStudio, not available in NV or other competing products, and an essential tool for AI tuning**), and supports interactive mouse-based analysis.
- **Result display**:

![performance%20analysis](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/performance%20analysis.png)

#### Memory Analysis

##### **Analyzing System Memory Changes Through Profiling and Call Stacks**

- **Operation**: Enable the call stack and memory view features when collecting data.
- **Value**: Observe the framework and CANN memory allocation and release status, and trace back to the frontend Python code with the call stack.
- **Result**: Analyze memory changes with the call stack. The result is shown as follows:

![in-memory%20analytics](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/in-memory%20analytics.gif)

##### **Using the msleaks Tool for In-Depth Memory Analysis**

- **Procedure**: Refer to the [msleaks tool usage guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/msleaks/atlas_msleaks_0001.html).
- **Value**: You can view the line chart of total framework memory allocation or the memory block diagram, and directly map them to the call stack, enabling in-depth analysis of framework memory usage.
- **Result**:

![msleaks](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/msleaks.gif)

## Performance Analysis Example

For specific performance analysis, enable **level1** profiling; otherwise, key operator information is missing.

### 1. Host Bound Diagnosis

Host bound refers to a situation where the total CPU task volume exceeds that of the NPU, causing the NPU to experience idle bubbles during execution. You can determine this by checking the synchronization lines of Host2Device. If the lines are skewed, it indicates that the set signal here occurs earlier than the wait signal, and the NPU executes as soon as it is ready, which is also device bound:

![host_bound_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_1.png)

If the issue is confirmed to be host bound, you can examine the CPU side to identify the dispatch time of each operator. When doing so, note that you need to find the cumulative CPU time across all operators, not just a single layer, because the first call takes a long time. For example, in the GmmSwigluQuant shown in the following figure, the first call on the CPU takes 1 ms, while each subsequent call takes only 200 μs.

![host_bound_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/host_bound_2.png)

At this point, some operators are carrying a heavy load, while others are holding the system back, and the latter outnumber the former. We prioritize **identifying the top operators whose host time exceeds device time, as these are the ones dragging performance down**, and they can be handed over to the operator team for focused analysis.

### 2. Network Topology Rationality Analysis

Sometimes, the model architecture is not implemented in the most efficient way, which is easy to identify in profiling. The following introduces the analysis approach and provides examples.

Generally speaking, the major hot operators in an LLM are the matrix multiplication computations in Attention and FFN. Together, these can account for more than 70% of the computation time in prefill and more than 50% in decode. If the overall time proportion does not meet expectations, if profiling reveals some new operators, or if there are too many concatenation-type operators, it is worth analyzing the model network to determine whether the operators are used incorrectly. In particular, concatenation-type operators deserve individual analysis.

For concatenation-type operators such as slice, split, and concat, as well as conversion operators like transpose and cast, their presence often results from the preceding and following operators not being directly compatible. If the preceding operator can directly perform the final processing on the output, it often saves the startup overhead of one operator and one redundant read/write. However, such a change may not always align with the basic design principles of the operator.

Here is a positive example. For a Matmul whose output shape is [m, n0 + n1], we connect two slices after it. Both slices take this [m, n0 + n1] tensor as input, and their outputs are [m, n0] and [m, n1], respectively. The first optimization idea is to replace the two slices with a single split, which can roughly halve the time consumption and release the [m, n0 + n1] device memory as early as possible. A further optimization idea is to split the matrix multiplication weight from [k, n0 + n1] into [k, n0] and [k, n1], and divide the original matrix multiplication task into two tasks (provided that the total time consumption of the two tasks is not much worse than before, and the kernel-splitting strategy does not cause issues), thereby completely eliminating this slice/split operation.

![network_1](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_1.png)

Here is a counterexample: Rmsnorm(fp16) + Cast(fp16->fp32) + Matmul(fp32). Although the inputs and outputs of Rmsnorm are both fp16, the internal computation uses fp32 for accuracy in accumulation. If you fuse the Cast into Rmsnorm, the Rmsnorm, which already uses fp32 internally, can omit a final fp32->fp16 cast. Together with the Cast we removed, this saves two casts and avoids one precision loss. Although this seems to achieve both precision and performance, an Rmsnorm with fp16 input and fp32 output violates the principle that the core input and output must be the same data type. Unless we can frequently find such structures in widely used open-source models and prove their universality, the operator team does not allow creating such operators.

![network_2](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/network_2.png)

### 3. Initial Diagnosis of Operator Performance

Use `"./ASCEND_PROFILER_OUTPUT/operator_details.csv"` for analysis to determine whether an operator has performance issues.

The Profiling tool calculates the average busy time of these pipelines on different cores (xxx_time), divides it by the complete kernel duration of the slowest core (task_duration), and obtains the pipeline utilization (xxx_ratio). Although these pipelines depend on each other and the transfer pipelines compete for bandwidth, operators can mask each other if they are designed properly. Therefore, we can preliminarily assume that **when the execution time of an operator is large enough, the operator should form a bound on a certain pipeline**, that is, the utilization should be high enough. Empirically, when a single operator takes 50μs, it can be considered that the operator should be on the bound pipeline, achieving a utilization of over 80%.

The following figure shows an example. The first row is an FA operator, and the second row is a Matmul operator. FA achieves 88.1% utilization on the vec pipeline, and the Matmul operator achieves 89.8% utilization on the mac pipeline. Their performance can be considered qualified.

![Operator%20performance](https://github.com/chengminhua/verl_data/raw/main/MindStudio_Insight_use/Operator%20performance.png)

### 4. Affinity Shape Adjustment

For a given model, hyperparameters are beyond our control. However, we can adjust factors such as concurrency, weight format, and sharding strategy to accommodate the operators and maximize their performance. This section focuses on two aspects: operator transfer efficiency and load balancing, and discusses the adjustment directions worth trying on the model side.

#### 4.1 Shape Optimized for Data Transfer Efficiency

MTE2 is a pipeline whose efficiency is heavily affected by shape. To ensure that MTE2 achieves maximum transfer efficiency, you need to satisfy at least one of the following two conditions:

**(1) The matrix being moved uses nz as its format (optimal)
(2) The tail axis of the matrix being moved is 512B-aligned and is not an integer multiple of 16KB (near-optimal)**

For weight matrices, during inference, especially in the decode phase, we usually satisfy condition (1), while during training, we usually satisfy condition (2). **If we cannot achieve condition (1), we must accommodate condition (2)**. Typical methods include:

1. If condition (1) is not met and the first axis of the matrix is friendly while the last axis is not, transpose it.
2. Adjust the TP sharding strategy to avoid an unfriendly last axis.

#### 4.2 Load-Balancing Affinity Shapes

When the operator shape is small, due to the operator semantics, we may not be able to utilize all cores, or even if all cores are enabled, load balancing may be poor. This section mainly analyzes small shapes in the decode phase.

First, determine the number of cores on the current NPU. If you are unsure, and the profiling results show numbers such as 20 or 40, the NPU has 20 cores. Otherwise, it has 24 cores. Here, a 24-core configuration actually represents a group consisting of one cube and two vectors. You can think of the cube as the primary core, with two vectors as secondary cores. If an operator is a pure vector operator, the concept of a group no longer applies. Instead, 40 or 48 vector cores act as primary cores and independently fetch logical tasks.

For the vector operator in LLMs, a common kernel-splitting strategy is to split along the highest dimension, that is, the batch dimension. This approach is common for operators with reduction operations on low dimensions (also called trailing axes), such as norm and dynamic quantization operators. Another strategy is to flatten the entire tensor, which allows operators to be split very finely, such as elementwise operators. For the first strategy, you can focus on load balancing at the model side. For example, if you run a batch of 48, but the hardware has 40 vector cores, these 40 cores loop twice, and in the second loop, most cores have no work to do. In this case, the batch size can be considered unfriendly. If you set the batch to 64 or 80, the performance is predictably lossless. Similarly, if the card has 48 cores, you can consider this batch size very friendly.

For cube-class operators, a common core-splitting strategy is to split M and N by base blocks (the K axis is an accumulation axis, and splitting it introduces determinism issues). The most common block sizes are baseM=128 and baseN=256. During the decode phase, the time is almost entirely spent moving weights, because the activation M is extremely small, and the M direction is likely split into only one block, so the right matrix only needs to be loaded once. Therefore, within the range of M≤128, you can increase M freely with almost no performance loss. If M is greater than 128, you can consider (128, 256] as the next performance tier.

In addition to M, the task of splitting the N axis also affects operator affinity. For example, in the MLA preprocessing in DeepSeek R1, the same activation (with shape [batch_size, 7168]) is used to perform matrix multiplications with two weights (with shapes [7168, 1536] and [7168, 576]). When batch_size is not large, even if baseN is reduced to 128, the N axis cannot fully utilize the number of cores. Therefore, the time for each of these two matrix multiplications is approximately equal to the time for a matrix multiplication that concatenates the two weights along the N axis (with shape [7168, 2112]). If you only consider model competitiveness, you would prefer to merge these two weights; otherwise, the bandwidth utilization of both small matrix multiplications will be very poor.

For the Attention operator, the common kernel splitting strategies are `q_seqlen`, `batch_size`, and `kv_headnum`. In the incremental phase, `q_seqlen` is merged by the MTP and GQA multiples, but it usually does not exceed 128, so a second task cannot be split out. In that case, the parallelism is basically `batch_size * kv_headnum`.

In general, we can identify whether an operator has load balancing issues based on its shape information and operator category, which helps us predict the optimal sharding strategy and the batch strategy that achieves the highest throughput.
