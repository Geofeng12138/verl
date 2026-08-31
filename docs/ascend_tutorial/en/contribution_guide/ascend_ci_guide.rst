NPU-CI Addition Guide
=======================

Last updated: 02/02/2026.

We provide guidance on adding CI test cases based on Huawei Ascend devices in verl.

The verl repository uses GitHub Actions as its CI platform, with a layered testing architecture to ensure code quality and system stability.
The NPU-related workflows mainly include:

* ``npu_unit_test.yml``: Runs unit tests.
* Files ending with ``_ascend.yml``: Run end-to-end tests or special tests for Ascend NPU.

Guide for Adding New Test Cases
-----------------------------------

1. Datasets and Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Weights and absolute paths on the pipeline machine:

+---------------------------------------+-------------------------------------------------------------------+
| Model name                            | Absolute path                                                     |
+=======================================+===================================================================+
| Qwen2.5-0.5B                          | ``${HOME}/.cache/models/Qwen/Qwen2.5-0.5B``                       |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen2.5-0.5B-Instruct                 | ``${HOME}/.cache/models/Qwen/Qwen2.5-0.5B-Instruct``              |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen2.5-1.5B-Instruct                 | ``${HOME}/.cache/models/Qwen/Qwen2.5-1.5B-Instruct``              |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen2.5-7B-Instruct                   | ``${HOME}/.cache/models/Qwen/Qwen2.5-7B-Instruct``                |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen2.5-VL-3B-Instruct                | ``${HOME}/.cache/models/Qwen/Qwen2.5-VL-3B-Instruct``             |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-0.6B                            | ``${HOME}/.cache/models/Qwen/Qwen3-0.6B``                         |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-8B                              | ``${HOME}/.cache/models/Qwen/Qwen3-8B``                           |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-8B-Base                         | ``${HOME}/.cache/models/Qwen/Qwen3-8B-Base``                      |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-30B-A3B-Instruct-2507           | ``${HOME}/.cache/models/Qwen/Qwen3-30B-A3B-Instruct-2507``        |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-32B                             | ``${HOME}/.cache/models/Qwen/Qwen3-32B``                          |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-VL-2B-Instruct                  | ``${HOME}/.cache/models/Qwen/Qwen3-VL-2B-Instruct``               |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-VL-4B-Instruct                  | ``${HOME}/.cache/models/Qwen/Qwen3-VL-4B-Instruct``               |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-4B-Instruct-2507                | ``${HOME}/.cache/models/Qwen/Qwen3-4B-Instruct-2507``             |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3-VL-8B-Instruct                  | ``${HOME}/.cache/models/Qwen/Qwen3-VL-8B-Instruct``               |
+---------------------------------------+-------------------------------------------------------------------+
| Skywork-Reward-V2-Llama-3.2-1B        | ``${HOME}/.cache/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B``  |
+---------------------------------------+-------------------------------------------------------------------+
| Qwen3.5-2B                            | ``${HOME}/.cache/models/Qwen/Qwen3.5-2B``                         |
+---------------------------------------+-------------------------------------------------------------------+

Datasets and absolute paths on the pipeline machine:

+--------------+---------------------------------------------------+
| Dataset Name | Absolute Path                                     |
+==============+===================================================+
| gsm8k        | ``${HOME}/.cache/datasets/openai/gsm8k``          |
+--------------+---------------------------------------------------+
| geo3k        | ``${HOME}/.cache/datasets/hiyouga/geometry3k``    |
+--------------+---------------------------------------------------+

**Note**

${HOME} is the root user's home directory.

For GPU cases, the weights are stored in the `~/models/` directory. To adapt, you can use a symbolic link: ``ln -s /root/.cache/models ~/models``

The following shows the original dataset. Process the data as needed, as shown in the example below.

   ``python examples/data_preprocess/gsm8k_multiturn_sft.py --local_dataset_path ${HOME}/.cache/datasets/openai/gsm8k``


2. Workflow YAML Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a new workflow, you can refer to the following template to create the ``.github/workflows/your_yml_ascend.yml`` file.

The main modifications include:

* Workflow name (``name``)
* Trigger condition (``on``)
* Runtime environment (``runs-on``)
* Container image (``container.image``)
* Specific execution steps (``jobs.<job_id>.steps``)

.. code-block:: yaml
   :linenos:

name: your_yml_ascend  # Unique identifier for the workflow
# Trigger condition configuration
on:
  push:
    branches:
      - main
      - v0.*
  pull_request:
    branches:
      - main
    paths:
      - ".github/workflows/your_yml_ascend.yml"  # Must include the path to this workflow file
      - "path/to/affected_files"               # Paths to related code that need monitoring

# Concurrency Control Strategy
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}  # Cancel in-progress jobs only for non-main branches

permissions:
  contents: read  # Principle of least privilege

jobs:
  your_job_name:  # Unique identifier for the job
    if: github.repository_owner == 'verl-project'  # Run only in the main repository
    runs-on: linux-aarch64-a2-4  # Hardware specification: A2 instance with 4 NPUs
    timeout-minutes: 60          # Job timeout threshold (in minutes)
    container:
      # Runtime image. This example uses the vLLM image.
      image: swr.ap-southeast-1.myhuaweicloud.com/base_image/ascend-ci/verl/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm
      options: >-
        --shm-size 16g  # Shared memory configuration
    env:
      HF_ENDPOINT: "https://hf-mirror.com"
      HF_HUB_ENABLE_HF_TRANSFER: "0"
    steps:
      - name: Check npu and CANN info
        run: |
          cat /usr/local/Ascend/ascend-toolkit/latest/"$(uname -i)"-linux/ascend_toolkit_install.info
          npu-smi info
      - name: Check initial pip list from image
        run: pip list
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0 
          clean: true 
      - name: Install dependencies
        run: |
          pip install --no-deps -e .
      - name: Verify environment
        run: pip list
      # The following are the specific test steps (customize them as needed)
      - name: Preprocess dataset
        run: python examples/data_preprocess/your_script.py --local_dataset_path ${HOME}/.cache/datasets/your_dataset
      - name: Execute NPU test
        run: |
          ray stop --force 
          bash tests/special_npu/your_test_script.sh

**Note**


The contents added to the ${HOME}/.cache/ folder are not deleted when the CI container is destroyed after the run. Avoid adding content to this folder.


3. Add unit tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Steps:

(1) Create or modify unit test files (for example, ``test_xxx.py``) in the ``tests/`` directory.
(2) If the test file path is not excluded by the ``--ignore-glob`` rule in ``npu_unit_test.yml``, it is automatically executed in the following commands:

   .. code-block:: yaml

      pytest -s -x --ignore-glob="xxx" --ignore-glob="xxx" tests/

(3) If the test path is within the ``--ignore-glob`` exclusion scope, add a step in ``npu_unit_test.yml`` to explicitly run the test.
(4) If you add a batch of related test cases, create a dedicated workflow file to keep things clear.

4. Add an end-to-end test script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Steps:

(1) Create the end-to-end test script in the ``tests/special_npu/`` directory.
(2) In the ``.github/workflows/`` directory, find the workflow file with the closest functionality that ends with ``_ascend.yml``, and add a step to call this script in it.
(3) If the test scenario is independent or complex, consider creating a separate new workflow file.

5. Recommended Testing Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Unit tests**: Cover core functions, classes, and methods to ensure correct logic.
* **Integration/end-to-end tests**: Cover typical training and inference pipelines to verify multi-module collaboration and hardware adaptation.
* **Resource management**: Multiple jobs in a workflow run in parallel. Set reasonable timeout values to prevent tasks from hanging for a long time. Keep the running time of a single job within 40 minutes.

Through the preceding steps, you can systematically add NPU-related automated tests to the verl repository, ensuring that code changes are fully validated before merging.
