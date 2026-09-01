Ascend Installation Guide
=========================

Last updated: 2026/08/13.

Key Updates
-----------

-  2026/08/03: Updated vLLM / vLLM-Ascend from ``0.18.0`` to ``0.23.0``\ , and synchronously adjusted the corresponding base environment versions for vLLM to torch ``2.10.0``\ , torch_npu ``2.10.0.post2``\ .
-  2026/05/13: Based on `PR
   #6291 <https://github.com/verl-project/verl/pull/6291>`__\ , vLLM updated vLLM /
   vLLM-Ascend from ``0.13.0`` to ``0.18.0``\ , and synchronously adjusted the corresponding base environment versions for vLLM to torch ``2.9.0``\ , torch_npu
   ``2.9.0.post2``\ .
-  2025/12/11: Existing verl scenarios currently support automatic identification of NPU device types. In principle, GPU
   scripts no longer require explicitly setting
   ``trainer.device=npu``\ when running on the Ascend NPU; new features can still prioritize specifying the device type by setting ``trainer.device``.

..

[Note] To automatically recognize the NPU device type, the environment running the program must include
 the ``torch_npu`` package. If the environment does not include ``torch_npu``\ , you must explicitly specify
 ``trainer.device=npu``\ .

Contents
--------

- `Hardware Support <#硬件支持>`_
- `Framework Backend Support Notes <#框架后端支持说明>`_
- `Deployment Guide <#部署指南>`_
   - `Docker Image Acquisition, Building, and Usage <#1-docker镜像获取构建和使用>`_
   - `Custom Installation - vLLM + FSDP/Megatron <#2-自定义安装-vllm--fsdpmegatron>`_
   - `Custom Installation - SGLang + FSDP/Megatron <#3-自定义安装-sglang--fsdpmegatron>`_
- `Appendix <#附录>`_

Hardware support
--------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3

`Ascend 950 series products <https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/get_start/install_guidance_A5.rst>`_


Framework Backend Support
-------------------------

The current NPU supports the deployment of the following common training and inference backends. You can directly obtain the released images using our `Ascend image instructions <dockerfile_build_guidance.rst>`__, or perform a custom installation according to the sections below.

.. list-table::
   :header-rows: 1

* - Inference engine
     - Training engine
   * - vLLM
     - FSDP/FSDP2/Megatron
   * - SGLang
     - FSDP/FSDP2/Megatron

Deployment Guide
--------

1. Obtaining, building, and using Docker images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can obtain the relevant images from `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, or build them from the Dockerfile yourself. For details, refer to
`Ascend Image Guide <dockerfile_build_guidance.rst>`__\ .


2. Custom installation - vLLM + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^

============= ======================================= ===================
Dependency    Version                                 Description
============= ======================================= ===================
HDK           ``26.0.rc1``                            NPU hardware driver and firmware
CANN          ``9.1.0``                               CANN software that helps developers develop and run AI applications on Ascend software and hardware platforms
Python        ``>=3.10, <3.13``\ , recommended ``3.12``
torch         ``2.10.0``                              PyTorch deep learning framework base package
torch_npu     ``2.10.0.post4``                        NPU PyTorch adaptation plugin
torchvision   ``0.25.0``                              PyTorch image processing library
torchaudio    ``2.10.0``                              PyTorch audio processing library
triton        ``3.5.0``                               Triton, used for writing custom operators
triton-ascend ``3.2.2``                               NPU Triton adaptation; for installation commands, refer to the `installation script <../../../../scripts/install_vllm_mcore_npu.sh>`_
transformers  ``5.10.4``                              Hugging Face large model library, providing model architectures and pre-trained weights
vLLM          ``0.23.0``                              High-performance LLM inference and serving engine
vLLM-Ascend   ``0.23.0``                              NPU vLLM backend adaptation
Megatron-LM   ``core_r0.16.0``                        Large-scale distributed training framework
MindSpeed     ``core_r0.16.0``                        Megatron-LM adaptation and optimization component on Ascend NPU
============= ======================================= ===================


Pre-installation Preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is a heterogeneous computing architecture on the NPU. The following are installation instructions for the A3 on the ARM platform. Follow the instructions below to download and install HDK and CANN,
or download and install from the `CANN community <https://www.hiascend.com/cann/download?versionId=723&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash

# Configure the user group
sudo groupadd HwHiAiUser
sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
# Install dependencies and configure the source
sudo yum makecache
sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
# Install the NPU driver
sudo yum install -y Atlas-A3-hdk-npu-driver-26.0.rc1
# Install the Toolkit. You can specify --install-path for a custom path
sudo yum install -y Ascend-cann-toolkit-9.1.0
sudo yum install -y Ascend-cann-A3-ops-9.1.0
# Verify after installation
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl;print(acl.get_soc_name())"

Source code installation
^^^^^^^^^^^^^^^^^^^^^^^^

We provide an `installation script <../../../../scripts/install_vllm_mcore_npu.sh>`_ for one-click deployment using conda. The script installs the environment step by step. If you encounter an installation error midway, check the cause based on the error message of the current step. Alternatively, leave us a message through an issue, and we will resolve it as soon as possible.

.. code:: bash

# Note: When installing on an x86 platform, configure an additional pip source using the following command:
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Enable the CANN environment. If you have customized the CANN path, modify the following enable commands based on your custom path.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-vllm-npu python=3.12 -y
   conda activate verl-vllm-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_vllm_mcore_npu.sh
   # If you only need to use the FSDP backend
   # USE_MEGATRON=0 bash verl/scripts/install_vllm_mcore_npu.sh

Filtering logs
^^^^^^^^^^^^^^^^^^^^^^^^
After you upgrade transformers to version 5.10.4, a large number of alias deprecation warnings may appear. You can add an environment variable to filter redundant logs.

.. code:: bash

   export TRANSFORMERS_VERBOSITY=error

3. Custom Installation-SGLang + FSDP/Megatron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^

============= ======================================= ===================
Dependency    Version                                 Description
============= ======================================= ===================
HDK           ``25.5.0``                              NPU hardware driver and firmware
CANN          ``>=8.5.0``                             CANN software, helping developers develop and run AI services on Ascend software and hardware platforms
Python        ``>=3.10, <3.12``\ , recommended ``3.11``
torch         ``2.8.0``                               PyTorch deep learning framework base package
torch_npu     ``2.8.0.post2``                         NPU PyTorch adaptation plugin
SGLang        ``v0.5.10``                             High-performance LLM inference engine
triton        ``3.5.0``                               Triton, used for writing custom operators
triton-ascend ``3.2.1``                               NPU Triton adaptation, for installation commands refer to the script `installation script <../../../../scripts/install_vllm_mcore_npu.sh>`_
transformers  ``5.3.0``                               Hugging Face large model library, providing model architectures and pre-trained weights
Megatron-LM   ``core_r0.16.0``                        Large-scale distributed training framework
MindSpeed     ``core_r0.16.0``                        Megatron-LM adaptation and optimization component on Ascend NPU
============= ======================================= ===================


Pre-installation Preparation (HDK & CANN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CANN is a heterogeneous computing architecture for NPUs. The following are the installation instructions for the A3 platform on ARM. Refer to the following instructions to download and install HDK and CANN. Alternatively, download and install them from the `CANN community <https://www.hiascend.com/cann/download?versionId=680&ids=d803%2Ch0501%2Ch0601%2Ch0702>`_ based on your system hardware model.

.. code:: bash

# Configure the user group
   sudo groupadd HwHiAiUser
   sudo useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   # Install dependencies and configure the repository
   sudo yum makecache
   sudo yum install -y gcc python3 python3-pip kernel-headers-$(uname -r) kernel-devel-$(uname -r) 
   sudo curl https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo && yum makecache
   # Install the NPU driver
   sudo yum install -y Atlas-A3-hdk-npu-driver-25.5.0
   # Install the Toolkit. You can specify --install-path to customize the path.
   sudo yum install -y Ascend-cann-toolkit-8.5.0
   sudo yum install -y Ascend-cann-A3-ops-8.5.0
   # Verify after installation
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   python3 -c "import acl;print(acl.get_soc_name())"

Source code installation
^^^^^^^^^^^^^^^^^^^^^^^^

We provide an `installation script <../../../../scripts/install_sglang_mcore_npu.sh>`_ for one-click deployment using conda. The script installs the environment step by step. If you encounter an installation error midway, check the cause based on the error message of the current step. Alternatively, submit an issue to us, and we will resolve it as soon as possible.

.. code:: bash

# Note: When installing on the x86 platform, configure an additional source for pip using the following command:
   # pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
   # Enable the CANN environment. If you have customized the CANN path, modify the following activation command based on your custom path.
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   conda create -n verl-sgl-npu python=3.11 -y
   conda activate verl-sgl-npu
   git clone --recursive https://github.com/verl-project/verl.git
   bash verl/scripts/install_sglang_mcore_npu.sh
   # If you only need to use the FSDP backend
   # USE_MEGATRON=0 bash verl/scripts/install_sglang_mcore_npu.sh

SGLang Usage Precautions
^^^^^^^^^^^^^^^

To support the SGLang backend on the current NPU, add the following environment variables:

.. code:: bash

# Support multi-process on a single NPU device
   export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
   export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# Work around the issue where Ray cannot identify device availability using the is_npu_available interface during device-side calls
   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# Define based on the current device and the required number of devices
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# in A3
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Required when enabling inference EP
   export SGLANG_DEEPEP_BF16_DISPATCH=1

Appendendix
----------------

Notes on ecosystem libraries not currently supported by the Ascend NPU
~~~~~~~~~~~~~~~~~~~~~~

The following ecosystem libraries are not currently supported on Ascend NPU in verl:

+------------------+--------------------------------------------------+
| Software         | Description                                      |
+==================+==================================================+
| ``flash_attn``   | Flash attention acceleration cannot be enabled  |
|                  | through a standalone ``flash_attn`` package.    |
|                  | It is supported through transformers.            |
+------------------+--------------------------------------------------+


