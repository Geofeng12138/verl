Ascend Installation Guide (A5)
=================================

Last updated: 08/03/2026.

Key Version Support and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
============= ================================================= ===================
Dependency    Version                                            Description                                                        
============= ================================================= ===================
CANN          Link to be updated after the Q2 CANN version is officially released   CANN software helps developers build and run AI workloads on Ascend software and hardware platforms
Python        ``3.11``                                          Python version                                                 
torch         ``2.10.0``                                        PyTorch deep learning framework base package                                 
torch_npu     Link to be updated after the Q2 torch_npu version is officially released   NPU PyTorch adaptation plugin                                       
triton        ``3.5.0``                                         Triton, used for writing custom operators                                 
triton-ascend ``3.2.2``                                         NPU Triton adaptation                                            
transformers  ``4.57.6``                                        Hugging Face large model library, providing model architectures and pretrained weights            
vLLM          ``0.23.0``                                        High-performance LLM inference and serving engine                                  
vLLM-Ascend   ``0.23.0``                                        NPU vLLM backend adaptation                                          
Megatron-LM   ``core_r0.12.0``                                  Large-scale distributed training framework                                       
MindSpeed     ``0c6c0ceaa523a96032dee1539a52032155e6404e``      Adaptation and optimization components for Megatron-LM on Ascend NPUs                  
============= ================================================= ===================

Environment Installation Steps
^^^^^^^^^^^^^^^^^

vLLM Inference Backend Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

# Install vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.23.0
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vllm-ascend
# Before installation, source the CANN environment: source /usr/local/Ascend/cann/set_env.sh
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout releases/v0.23.0
pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..


Megatron Training Backend Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source installation instructions for MindSpeed, Megatron, and their related dependencies:

.. code:: bash

    # MindSpeed
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    git checkout 0c6c0ceaa523a96032dee1539a52032155e6404e
    pip install -e .
    cd ..

    # Megatron
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.12.0
    pip install -e .
    cd ..

# Configure environment variables
export PYTHONPATH=$PYTHONPATH:your_path/Megatron-LM
export PYTHONPATH=$PYTHONPATH:your_path/MindSpeed

# Install mbridge
pip install mbridge

verl Dependency Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/verl-project/verl.git
    cd verl
    pip install -e .
    pip install -r requirements-npu.txt

