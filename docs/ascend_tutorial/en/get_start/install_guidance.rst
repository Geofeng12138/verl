Ascend Installation Guide
=================

Last updated: 2026/08/13.

Key Updates
--------------

-  2026/08/03: Updated vLLM / vLLM-Ascend from ``0.18.0`` to ``0.23.0``, and the corresponding base environment versions for vLLM were adjusted to torch ``2.10.0`` and torch_npu ``2.10.0.post2``.
-  2026/05/13: vLLM has been updated from ``0.13.0`` to ``0.18.0`` for vLLM / vLLM-Ascend as per `PR #6291 <https://github.com/verl-project/verl/pull/6291>`__, and the corresponding base environment versions for vLLM were adjusted to torch ``2.9.0`` and torch_npu ``2.9.0.post2``.
-  2025/12/11: The existing verl scenarios currently support automatic NPU device type detection. In principle, GPU scripts no longer need to explicitly set ``trainer.device=npu`` when running on Ascend; for new features, you can still specify the device type by setting ``trainer.device``.

..

[Description] Automatic NPU device type detection requires that the environment where the program runs includes the ``torch_npu`` software package. If the environment does not include ``torch_npu``, you still need to explicitly specify ``trainer.device=npu``.

Table of Contents
--------

- `Hardware Support <#hardware-support>`_
- `Framework Backend Support <#framework-backend-support>`_
- `Deployment Guide <#deployment-guide>`_
   - `Docker Image Acquisition, Build, and Usage <#1-docker-image-acquisition-build-and-usage>`_
   - `Custom Installation - vLLM + FSDP/Megatron <#2-custom-installation-vllm--fsdpmegatron>`_
   - `Custom Installation - SGLang + FSDP/Megatron <#3-custom-installation-sglang--fsdpmegatron>`_
- `Appendix <#appendix>`_

Hardware Support
==================

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3

`Ascend 950 series products <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/get_start/install_guidance_A5.rst>`_


Framework Backend Support Description
------------------------------------------

The following common training and inference backends are currently supported for deployment on the NPU. You can directly obtain the released images based on our `Ascend image description <dockerfile_build_guidance.rst>`__, or perform a custom installation as described below.

.. list-table::
   :header-rows: 1

* - Inference engine
  - Training engine
* - vLLM
  - FSDP/FSDP2/Megatron
* - SGLang
  - FSDP/FSDP2/Megatron

Deployment Guide
====================

1. Obtaining, Building, and Using the Docker Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can obtain the relevant image from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_ or build it yourself from the DockerFile. For related instructions, refer to the
`Ascend image description <dockerfile_build_guidance.rst>`__\ .


2. Custom Installation - vLLM + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============= ======================================= ===================
Dependency    Version                                Description
============= ======================================= ===================
HDK           ``26.0.rc1``                            NPU hardware driver and firmware
CANN          ``9.1.0``                               CANN software helps developers build and run AI workloads on the Ascend software and hardware platform
Python        ``>=3.10, <3.13``\ , recommended ``3.12``
torch         ``2.10.0``                              PyTorch deep learning framework base package
torch_npu     ``2.10.0.post4``                        NPU PyTorch adaptation plugin
torchvision   ``0.25.0``                              PyTorch image processing library
torchaudio    ``2.10.0``                              PyTorch audio processing library
triton        ``3.5.0``                               Triton, used for writing custom operators
triton-ascend ``3.2.2``                               NPU Triton adaptation. For installation commands, refer to the `installation script <../../../../scripts/install_vllm_mcore_npu.sh>`_
transformers  ``5.10.4``                              Hugging Face large model library, providing model architectures and pretrained weights
vLLM          ``0.23.0``                              High-performance LLM inference and serving engine
vLLM-Ascend   ``0.23.0``                              NPU vLLM backend adaptation
Megatron-LM   ``core_r0.16.0``                        Large-scale distributed training framework
MindSpeed     ``core_r0.16.0``                        Megatron-LM adaptation and optimization components on Ascend NPU
============= ======================================= ===================


Installation Preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^

CANN is the heterogeneous computing architecture on NPUs. The following are the installation commands for the A3 platform on the ARM architecture. Follow these commands to download and install HDK and CANN, or download and install them from the `CANN community <https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash

#Configure the user group
sudo groupadd HwHiAiUser
sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
# Install dependencies and configure the source
sudo yum makecache
sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
# Install the NPU driver
sudo yum install -y Atlas-A3-hdk-npu-driver-26.0.rc1
# Install the Toolkit. You can specify a custom path with --install-path.
sudo yum install -y Ascend-cann-toolkit-9.1.0
sudo yum install -y Ascend-cann-A3-ops-9.1.0
# Verify the installation
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl;print(acl.get_soc_name())"

Source Code Installation
^^^^^^^^^^^^^^^^^^^^^^^^

We provide a one-click deployment `installation script <../../../../scripts/install_vllm_mcore_npu.sh>`_ based on conda. The script installs the environment step by step. If an installation error occurs during the process, check the cause based on the error message of the current step, or leave us a message through an issue. We will resolve it as soon as possible.

.. code:: bash

# Note: When installing on an x86 platform, pip requires an additional source configuration. Use the following command:
# pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
# Enable the CANN environment. If you customized the CANN path, modify the following enable commands based on your custom path.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
conda create -n verl-vllm-npu python=3.12 -y
conda activate verl-vllm-npu
git clone --recursive https://github.com/verl-project/verl.git
bash verl/scripts/install_vllm_mcore_npu.sh
# If you only need to use the FSDP backend
# USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh

Log filtering
^^^^^^^^^^^^^^^^^^^^^^^^
After upgrading the transformers version to 5.10.4, a large number of alias deprecation warnings may appear. You can add an environment variable to filter redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

3. Custom Installation - SGLang + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

============= ======================================= ===================
Dependency    Version                                Description
============= ======================================= ===================
HDK           ``25.5.0``                              NPU hardware driver and firmware
CANN          ``>=8.5.0``                             CANN software helps developers develop and run AI workloads on the Ascend software and hardware platform
Python        ``>=3.10, <3.12``\ , recommended ``3.11``
torch         ``2.8.0``                               PyTorch deep learning framework base package
torch_npu     ``2.8.0.post2``                         NPU PyTorch adaptation plugin
SGLang        ``v0.5.10``                             High-performance LLM inference engine
triton        ``3.5.0``                               Triton, used for writing custom operators
triton-ascend ``3.2.1``                               NPU Triton adaptation. For installation commands, refer to the `installation script <../../../../scripts/install_vllm_mcore_npu.sh>`_
transformers  ``5.3.0``                               Hugging Face large model library, providing model architectures and pretrained weights
Megatron-LM   ``core_r0.16.0``                        Large-scale distributed training framework
MindSpeed     ``core_r0.16.0``                        Adaptation and optimization component for Megatron-LM on Ascend NPU
============= ======================================= ===================


Installation Preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^

CANN is the heterogeneous computing architecture on NPUs. The following are the installation commands for the A3 on the ARM platform. Follow these commands to download and install HDK and CANN, or download and install them from the `CANN community <https://www.hiascend.com/cann/download?versionId=680&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash

# Configure the user group
sudo groupadd HwHiAiUser
sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
# Install dependencies and configure the source
sudo yum makecache
sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
# Install the NPU driver
sudo yum install -y Atlas-A3-hdk-npu-driver-25.5.0
# Install the Toolkit. You can specify a custom path with --install-path
sudo yum install -y Ascend-cann-toolkit-8.5.0
sudo yum install -y Ascend-cann-A3-ops-8.5.0
# Verify the installation
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl;print(acl.get_soc_name())"

Source Code Installation
^^^^^^^^^^^^^^^^^^^^^^^^

We provide a one-click deployment `installation script <../../../../scripts/install_sglang_mcore_npu.sh>`_ based on conda. The script installs the environment step by step. If an installation error occurs during the process, check the cause based on the error message of the current step, or leave us a message through an issue, and we will resolve it as soon as possible.

.. code:: bash

# Note: When installing on an x86 platform, pip requires an additional source configuration. Use the following command:
# pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
# Enable the CANN environment. If you customized the CANN path, modify the following enable command based on your custom path.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
conda create -n verl-sgl-npu python=3.11 -y
conda activate verl-sgl-npu
git clone --recursive https://github.com/verl-project/verl.git
bash verl/scripts/install_sglang_mcore_npu.sh
# If you only need to use the FSDP backend
# USE_MEGATRON=0 bash verl/scripts/install_sglang_mcore_npu.sh

SGLang Usage Precautions
^^^^^^^^^^^^^^^^^^^^^^^

The following environment variables must be added to support the SGLang backend on the current NPU:

.. code:: bash

# Support for multiple processes on a single NPU
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# Work around Ray failing to identify device availability on the device side when it cannot use the is_npu_available interface
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# Define based on the current device and the required number of cards
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# in A3
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Required when enabling the inference EP
   export SGLANG_DEEPEP_BF16_DISPATCH=1

Appendix
----------------

Ascend Ecosystem Library Support Status
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In verl, the following ecosystem libraries are not yet supported on Ascend:

+------------------+--------------------------------------------------+
| Software         | Description                                      |
+==================+==================================================+
| ``flash_attn``   | Flash attention acceleration is not supported    |
|                  | through a standalone ``flash_attn`` package; it  |
|                  | is supported through transformers.               |
+------------------+--------------------------------------------------+


