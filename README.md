## NVIDIA CUDA GPU support for Docker/WSL

### The Hardware

I will describe here the hard way of getting NVIDIA drivers,
pytorch, AutoGPTQ, urllib3 and many other stuff to work under
Windows Subsystem for Linux v2, where I was running tests.
But on bare metal or ML/GPU cloud intances gets easier.

My hardware was an ASUS ROG Strix G713RW laptop with:

* AMD Ryzen 9 6900HX 32GB DDR5 with Radeon Graphics.
* NVIDIA GeForce RTX 3070 Ti 8GB GDDR6 Laptop Edition. 

The complications are:

* Host OS Windows 11 Pro 64 bits (AMD):
  * Windows Virtulization Platform + WSL v2 features enabled.
  * Device Security -> Core Isolation -> Memory Integraty -> Off.
  * NVIDIA Driver Version 560.94 supports Direct 3D 12.1.
* Guest Operating System Ubuntu 22.04 x86-64 (not AMD-64):
  * CUDA Driver Version = 12.6 (Installed on Linux from NVIDIA site).

if your are going to use GPU power, then this has to be
configured manually (I'm too lazy to *Makify* it).

## WSL v2 increasing RAM and Swap

To increase the RAM and SWAP memory on Windows Subsystem for Linux v2:
```powershell
# https://learn.microsoft.com/en-us/windows/wsl/wsl-config

# As Local User
Start-Process -File notepad.exe -ArgumentList "$env:userprofile/.wslconfig"

# Content of .wslconfig:
# [wsl2]
# memory=25GB
# swap=25GB

# Stop the VM
wsl --shutdown

# As Local Administrator
Restart-Service LxssManager
```

### Installation of NVIDIA CUDA Drivers (Binaries)

General documentation is available here for you to read:

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

After following the install steps, you can check if works.

### Pre-flight checks on the Linux WSL Guest.

Check first what is already built-in in the WSL Linux image:

```bash
nvidia-smi
```

Output should be like this (python3.10 is the running the Chatbot):
```bash
Fri Sep 27 23:44:00 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.02              Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   57C    P8             16W /  130W |    6053MiB /   8192MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     75869      C   /python3.10                                 N/A      |
+-----------------------------------------------------------------------------------------+
```

### Installation of NVIDIA CUDA Driver Libraries (From Source Code)

This is needed so Python (pip) can compile the necesary ML packages 
for your CUDA Architecture:

```bash
###
# Downloads/Documentation:
# https://developer.nvidia.com/cuda-downloads (Linux > Installer Type > deb(network))
# https://developer.nvidia.com/cuda-toolkit-archive (For older version, incl. docs.)
# Tested:
# CUDA 12.6 - 
# CUDA 12.1 - Not supported by PyTorch (Aug/2023) breaks AutoGPTQ CUDA ext. compilation.
# CUDA 11.8 - Compiles with PyTorch / AutoGPTQ and my works with my RTX 3070.
###
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

```bash
## Test the CUDA code compilation
git clone https://github.com/nvidia/cuda-samples
cd cuda-samples/Samples/1_Utilities/deviceQuery
make  # It must compile for your GPU natively, no GCC flags
./deviceQuery
(...)
Device 0: "NVIDIA GeForce RTX 3070 Ti Laptop GPU"
  CUDA Driver Version / Runtime Version          12.6 / 12.6
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 8192 MBytes (8589410304 bytes)
  (046) Multiprocessors, (128) CUDA Cores/MP:    5888 CUDA Cores
(...)
```

### Install NVIDIA Container Toolkit (Docker Daemon Settings)

General documentation is available here for you to read:

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation

After following the install steps, you can check if works.


You can play a bit with the NVIDIA Container Toolkit (If you have docker):
```
sudo apt-get install -y nvidia-docker2
sudo docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```
Output should be like this:
```bash
(...)
GPU Device 0: "Ampere" with compute capability 8.6
> Compute 8.6 CUDA device: [NVIDIA GeForce RTX 3070 Ti Laptop GPU]
47104 bodies, total time for 10 iterations: 48.482 ms
= 457.649 billion interactions per second
= 9152.976 single-precision GFLOP/s at 20 flops per interaction
```

This is the recommended way to install the NVIDIA Container Toolkit
for Docker (Makes changes to `/etc/docker/daemon.json`), so you can
run GPU accelerated containers:
```bash
# Reference: https://hub.docker.com/r/ollama/ollama
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### NVIDIA CUDA on Windows Subsystem for Linux v2 (aka WSL2):

* https://developer.nvidia.com/cuda/wsl
* https://developer.nvidia.com/cuda-downloads
* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
* https://documentation.ubuntu.com/wsl/en/latest/tutorials/gpu-cuda/
* https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl
* https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute
