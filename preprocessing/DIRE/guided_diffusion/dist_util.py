"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist
from . import logger
# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

class COMM_WORLD:
    def __init__(self, size = 1, rank = 0):
        self.size = size
        self.rank = rank
        
    def Get_rank(self):
        return self.rank
    
    def Get_size(self):
        return 1

class MPI:
    def __init__(self, size = 1, rank = 0):
        self.COMM_WORLD = COMM_WORLD(size, rank)



MPI = MPI()


def setup_dist(devices=""):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    devices_list = [] 
    for device in devices.split(","):
        devices_list.append(int(device))
    # print(devices_list)
    GPUS_PER_NODE = len(devices_list)
    # print("GPUS_PER_NODE: ",GPUS_PER_NODE)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{devices_list[MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE]}"
    # print(f"rank: {MPI.COMM_WORLD.Get_rank()}, {os.environ['CUDA_VISIBLE_DEVICES']}")
    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """

    with bf.BlobFile(path, "rb") as f:
        data = f.read()

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
