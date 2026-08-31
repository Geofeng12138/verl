# Model Evaluation

Last updated: 07/14/2026.

The steps are the same across different models; Qwen3-30B is used as an example.

We use AISBench to evaluate models. This tool supports evaluation across multiple inference backends, including vLLM and SGLang.

## 1. Installation Methods

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~


## 2. Download the evaluation dataset

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~

## 3. Weight Conversion

verl currently supports saving Hugging Face format model weights directly through mbridge, which can be used without conversion.

If the model weights are not in the Hugging Face (HF) format, convert them to the HF format before evaluation.

For the conversion method, refer to the native verl [conversion method](../../../../../docs/advance/checkpoint.rst).

## 4. vLLM Inference Evaluation

**Start the vLLM server service**

Start the inference server with the following command. The parameters you need to modify are `model` and `tensor-parallel-size`.

model: the path to the Hugging Face model after weight conversion from the trained weights.

tensor-parallel-size: The number of tensor parallel replicas. It is recommended that the TP configuration remain consistent with the inference configuration used during training.

data-parallel-size: The number of data parallel replicas. For DP, it is recommended to keep this consistent with the configuration used during training inference. The default value is 1.

port: You can set any available port.

~~~bash
vllm serve /path/to/Qwen3-30B/ \
       --served-model-name auto \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 24 \
       --max-model-len 22528 \
       --max-num-batched-tokens 22528 \
       --enforce-eager \
       --trust-remote-code \
       --distributed_executor_backend=mp \
       --tensor-parallel-size 8 \
       --data-parallel-size 1 \
       --generation-config vllm \
       --port 8080
~~~

**Modify the AISBench inference configuration to start the vllm_client evaluation**

Open the inference configuration file benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py.

host_port must match the port on the server side. Modify max_seq_len and max_out_len according to the model configuration.
~~~bash
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
~~~

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
~~~

Open another window for evaluation and run the evaluation command:
~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~
## 5. SGLang Inference Evaluation
For evaluation, refer to the "Evaluation" section in the [Ascend SGLang Best Practices](../../model_support/examples/ascend_sglang_best_practices.rst).