Multi-Machine Task Startup Operation Guide
===================================

Last updated: 07/28/2026.

Introduction
----------------------------------

In large-scale model training scenarios, a single machine often cannot meet the computing power requirements, so multi-machine collaborative training is needed. verl implements distributed scheduling based on the Ray framework. Developers need to correctly start a Ray cluster on multiple nodes and configure the Ascend NPU-related environment variables to successfully launch multi-machine training tasks.

This article helps developers understand the following:

1. Prerequisites
2. Multi-machine Task Startup

Prerequisites
-----------------------------------

1. Environment and Network Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before multi-machine training, ensure that all nodes meet the following conditions:

- Each node has completed environment setup as described in the `Ascend Installation Guide <../../get_start/install_guidance.rst>`_, and the versions of key components such as verl, Ray, PyTorch, torch-npu, and CANN are consistent
- The training network segments between nodes are interconnected, and the Ray port, Dashboard port, and the HCCL port range configured later are accessible. ``ping`` only verifies basic connectivity; if the cluster has a firewall enabled, you must also confirm that the TCP ports are not blocked
- The training script paths and model/data/checkpoint paths on each node are consistent (a shared file system such as NFS is recommended)
- Each node has completed the installation of the NPU driver and CANN software stack, and ``npu-smi info`` can properly identify the devices
- The system time on each node is kept synchronized as much as possible to avoid timeline confusion during log and task troubleshooting

2. Obtain the communication NIC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On each node, first check the available network interfaces and their IPv4 addresses, because multi-machine communication depends on correct network interface configuration:

.. code-block:: bash

  ip -o -4 addr show scope global | awk '{print $2, $4}'

Select the network interface card (NIC) used for multi-machine training communication, and record the NIC name corresponding to each node. If the primary node IP is already determined, you can also run the following command on each node to check the NIC used to access the primary node:

.. code-block:: bash

  MASTER_ADDR="IP FOR MASTER NODE"
  ip route get "$MASTER_ADDR" | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}'

In subsequent configurations, ``HCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, and ``SOCKET_IFNAME`` in the startup script should use this network interface name.

3. Confirm the node role
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A multi-machine cluster contains one **master node (Master)** and several **worker nodes (Worker)**:

- **Main node**: Starts the Ray Head service, manages cluster scheduling, and triggers the training task after all child nodes have joined.
- **Child node**: Registers with the main node, joins the Ray cluster, and then waits for task scheduling.

Select one of the nodes as the primary node, and record its IP address.

Multi-machine task startup
-----------------------------------

1. Environment Variable Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure the following environment variables on **all nodes**:

.. code-block:: bash

# Ray log deduplication and detailed error output
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

# Ascend NPU dispatch optimization, set to 1 for graph mode, set to 2 for non-graph mode
export TASK_QUEUE_ENABLE=1

# HCCL communication timeout configuration (unit: seconds). Increase the value appropriately based on the model size.
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600

# HCCL port range configuration to avoid port conflicts
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# NPU visible device configuration
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Communication NIC configuration; replace with the actual NIC name of the current node
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

# File descriptor limit
ulimit -n 32768

# Optional Configuration
# Disable Hugging Face asynchronous weight loading to avoid excessive host memory peaks during the model loading phase in some environments
export HF_DEACTIVATE_ASYNC_LOAD=1

2. Write the multi-machine startup script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following script can be executed uniformly on all nodes. The script automatically determines the primary/child node role based on the current node IP:

.. code-block:: bash

# Clean up Ray processes that may remain from the previous training
pkill -9 python
ray stop --force
rm -rf /tmp/ray

# ====== Configuration that you need to modify ======
# Training script path
DEFAULT_SH="./run_*.sh"
echo "Use $DEFAULT_SH"

# Number of nodes and NPUs per node
NNODES=2
NPUS_PER_NODE=16

# Master node IP
MASTER_ADDR="IP FOR MASTER NODE"

# Network interface card of the current node
SOCKET_IFNAME="Your SOCKET IFNAME"
# ====== End of configuration ======

# Get the current node IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # ====== Main node ======
    ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'
```

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
            ray status
            bash $DEFAULT_SH
            break
        else
            echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
            sleep 5
        fi
    done
  else
    # ====== Worker nodes ======
    while true; do
        ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

        ray status
        if [ $? -eq 0 ]; then
            echo "Successfully connected to the Ray cluster!"
            break
        else
            echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
            sleep 5
        fi
    done
  fi

  sleep 600

**Script Configuration Parameter Description:**

.. list-table::
   :header-rows: 1

* - Parameter
  - Description
* - ``DEFAULT_SH``
  - The path to the sh configuration file used for training, for example, ``run_qwen3moe-30b_grpo_megatron_vllm_npu.sh``
* - ``NNODES``
  - The number of nodes participating in training
* - ``NPUS_PER_NODE``
  - The number of NPUs per node, for example, 16 for Atlas 800T A3
* - ``MASTER_ADDR``
  - The IP address of the master node. This parameter must be the same on all nodes
* - ``SOCKET_IFNAME``
  - The communication network interface name of the current node. It may differ across nodes

3. Start Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the script above as ``ray_start.sh`` and run it on **all nodes** separately:

.. code-block:: bash

  bash ray_start.sh

Suggested execution order:

1. Start the script on the **primary node** first, and wait for the Ray Head service to be ready.
2. Then start the script on each **secondary node**; the secondary nodes automatically register with the primary node.
3. After the primary node detects that all nodes have joined, it automatically triggers the training task.

4. Monitoring the training status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After training starts, you can monitor it in the following ways:

**Ray Dashboard**

Open a browser and visit ``http://<MASTER_ADDR>:8260`` to view the Ray cluster status, resource usage, and task running status.

**Command Line View**

.. code-block:: bash

  ray status

**Training Logs**

The location of the training log output depends on the training script that ``DEFAULT_SH`` points to. If the training script is configured with a log file, use the following command to view it in real time:

.. code-block:: bash

  tail -f <TRAINING_LOG_PATH>
