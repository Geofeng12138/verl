Ascend Image Description
===================================

Last updated: 08/10/2026.


Image Acquisition and Public Image Addresses
--------------------------------------------------

Ascend hosts daily-built A2/A3 images in `quay.io/ascend/verl <https://quay.io/repository/ascend/verl?tab=tags&tag=latest>`_, which are built from the `Dockerfile <../../../../docker/ascend>`_. For details, see :ref:`Dockerfile image build script list <ascend-dockerfile-list>`.

The daily build image name format is: latest-{inference backend}-{applicable product information}-{operating system}-{other fields}

verl release image name format: {verl release version}-{CANN version}-{TorchNPU version}[-{applicable product information}-{operating system}]-{Python version}[-{inference backend}-{other fields}]



Mirror Hardware Support
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


Latest image component version information list
----------------

================= ============
Component          Version
================= ============
Base image         Ubuntu 22.04
Python             3.12
CANN               9.1.0
torch              2.10.0
torch_npu          2.10.0.post4
torchvision        0.25.0
vLLM               0.23.0
vLLM-ascend        0.23.0
Megatron-LM        core_r0.16.0
MindSpeed          core_r0.16.0
triton-ascend      3.2.2
mbridge            0.15.1
SGLang             v0.5.10
sgl-kernel-npu     2026.02.01
================= ============



.. _ascend-dockerfile-list:

Dockerfile Build Image Script Checklist
-------------------------------------------------

**General-purpose image**

============== ==================== ============== ==============================================================
Device type       CANN base image version      Inference backend        Reference file
============== ==================== ============== ==============================================================
A2              9.1.0                  vLLM            `Dockerfile.ascend_9.1.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a2>`_
A3              9.1.0                  vLLM            `Dockerfile.ascend_9.1.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.1.0_a3>`_
A2              8.5.0                  vLLM            `Dockerfile.ascend_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2>`_
A3              8.5.0                  vLLM            `Dockerfile.ascend_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3>`_
A2              8.5.0                  SGLang          `Dockerfile.ascend.sglang_8.5.0_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a2>`_
A3              8.5.0                  SGLang          `Dockerfile.ascend.sglang_8.5.0_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.5.0_a3>`_
A2              8.3.RC1                vLLM            `Dockerfile.ascend_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2>`_
A3              8.3.RC1                vLLM            `Dockerfile.ascend_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3>`_
A2              8.3.RC1                SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a2>`_
A3              8.3.RC1                SGLang          `Dockerfile.ascend.sglang_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend.sglang_8.3.rc1_a3>`_
A2              8.2.RC1                vLLM            `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
A3              8.2.RC1                vLLM            `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_
============== ==================== ============== ==============================================================


**verl release version image**

============== ==================== ============== ============== ==============================================================
Device type      CANN base image version   Inference backend   verl version   Reference file                                
============== ==================== ============== ============== ==============================================================
A2              9.0.0                vLLM          release/v0.8.0 `Dockerfile.ascend_9.0.0_a2_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a2_v0.8.0>`_     
A3              9.0.0                vLLM          release/v0.8.0 `Dockerfile.ascend_9.0.0_a3_v0.8.0 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_9.0.0_a3_v0.8.0>`_ 
A2              8.5.0                vLLM          release/v0.7.1 `Dockerfile.ascend_8.5.0_a2_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a2_v0.7.1>`_     
A3              8.5.0                vLLM          release/v0.7.1 `Dockerfile.ascend_8.5.0_a3_v0.7.1 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.0_a3_v0.7.1>`_ 
============== ==================== ============== ============== ==============================================================


**Custom Model Image**

============== ==================== ============== ============== ==============================================================
Device type      CANN base image version   Inference backend   Model      Reference file                            
============== ==================== ============== ============== ==============================================================
A2              8.5.2                vLLM          Qwen3.5        `Dockerfile.ascend_8.5.2_a2_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a2_qwen3-5>`_   
A3              8.5.2                vLLM          Qwen3.5        `Dockerfile.ascend_8.5.2_a3_qwen3-5 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.5.2_a3_qwen3-5>`_ 
============== ==================== ============== ============== ==============================================================



**Description:**

* In the image with ``vLLM`` as the inference backend, vLLM, vLLM-ascend, MindSpeed, Megatron-LM, and verl are installed from source, with the source code located in the image root directory ``/``.
* In the image with ``SGLang`` as the inference backend, SGLang, MindSpeed, and verl are installed from source, with the source code located in the image root directory ``/``.


Mirror build command example
--------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd {verl-root-path}/docker/ascend

   # Build the image
   # vLLM
   docker build -f Dockerfile.ascend_8.5.0_a2 -t verl-ascend:8.5.0-a2 .
   # SGLang
   docker build -f Dockerfile.ascend.sglang_8.5.0_a2 -t verl-ascend-sglang:8.5.0-a2 .

   # Query local images after build
   docker images

**Description:**

* Using the vLLM image as an example, ``Dockerfile.ascend_8.5.0_a2`` is the Dockerfile file name. In ``verl-ascend:8.5.0-a2``, verl-ascend is the custom image name, and 8.5.0-a2 is the custom image tag.

Container Startup Command Template
----------------

.. code:: bash

   docker run -dit \
       --ipc=host \
       --network host \
       --name {your_docker_name} \
       --privileged \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
       -v /usr/local/sbin:/usr/local/sbin \
       -v /usr/sbin:/usr/sbin \
       -v /home:/home \
       -v /data:/data \
       {image_name}:{tag} \
       /bin/bash

**Description:**

* If you need to mount other local paths to the container, add ``-v <host machine path>:<path in container>`` yourself.
* It is recommended that you replace ``{your_docker_name}`` with a meaningful container name.
* The ``--privileged`` parameter grants the container extended permissions; evaluate whether it is necessary based on your actual security requirements.
* Replace ``{image_name}:{tag}`` with the image name and tag that correspond to the container build.

Start the container
--------------------

.. code:: bash

   docker start {your_docker_name}

Entering a Running Container
------------------

.. code:: bash

   docker exec -it {your_docker_name} bash


Declaration
--------------------
The Ascend-related Dockerfiles and images provided in verl are reference samples for trial and evaluation purposes. If you plan to use them in a production environment, contact us through official channels. Thank you.
