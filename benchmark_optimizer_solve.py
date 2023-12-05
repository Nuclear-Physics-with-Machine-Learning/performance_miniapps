import sys, os
import pathlib
import argparse

import jax
import jax.numpy as numpy

# Pull a function to set the compute parameters and device
from utils.startup import set_compute_parameters, should_do_io
from utils.mpi import init_mpi, discover_local_rank


# Create very basic argument parsing:
parser = argparse.ArgumentParser("JAX_QMC miniapp benchmark")
parser.add_argument("-d", "--data-directory",
                    type=pathlib.Path,
                    required=True,
                    help="Top level directory of input data.")
parser.add_argument("-s", "--solver",
                    type=lambda x : str(x).lower(),
                    default="cg",
                    choices=["cholesky", "conjugategradient", "cg"])
parser.add_argument("-i", "--iterations",
                    type=int,
                    default=25, required=False,
                    help="Number of iterations to use as a measurement")
parser.add_argument("-p", "--precision", 
                    type = str,
                    choices = ["float32", "bfloat16", "float64"],
                    default="float64", required=False,
                    help = "Precision of the optimization routine")

args = parser.parse_args()


from utils.optimizer import Solver
if args.solver == "cholesky":
    args.solver = Solver.Cholesky
elif args.solver == "cg" or args.solver == "conjugategradient":
    args.solver = Solver.ConjugateGradient


# # Initialize mpi4py:
MPI_AVAILABLE, rank, size = init_mpi(distributed=True)
local_rank = discover_local_rank()

target_device = set_compute_parameters(local_rank)
# MPI_AVAILABLE = False
# rank = 0
# size = 1
# local_rank = 0

# target_device="cpu:0"

# First, on every rank, read the data needed:
# It is stored in a folder full of ranks

data_file = args.data_directory / pathlib.Path(f"rank_{rank}/step_4.npz")


this_ranks_npz = numpy.load(str(data_file), allow_pickle=True)

# Unpack the objects from the file that are needed and move them to the appropriate device:

dpsi_i    = jax.device_put( numpy.asarray(this_ranks_npz['dpsi_i'], dtype=args.precision),    target_device)
dpsi_i_EL = jax.device_put( numpy.asarray(this_ranks_npz['dpsi_i_EL'], dtype=args.precision), target_device)
energy    = jax.device_put( numpy.asarray(this_ranks_npz['energy'], dtype=args.precision),    target_device)
# x         = jax.device_put( numpy.asarray(this_ranks_npz['x'], dtype=args.precision),         target_device)
# spin      = jax.device_put( numpy.asarray(this_ranks_npz['spin'], dtype=args.precision),      target_device)
# isospin   = jax.device_put( numpy.asarray(this_ranks_npz['isospin'], dtype=args.precision),   target_device)
jacobian  = jax.device_put( numpy.asarray(this_ranks_npz['jacobian'], dtype=args.precision),  target_device)

w_params = this_ranks_npz["w_params"]

# We need to know the total number of walkers in this run.
# Easiest way is to infer it from data:
n_walkers = jacobian.shape[0]*size

args.n_walkers = n_walkers

# For these two, need to do some tree mapping:
from jax.tree_util import tree_map
w_params  = tree_map(lambda x : jax.device_put(numpy.asarray(x)), this_ranks_npz['w_params'].item())
opt_state  = tree_map(lambda x : jax.device_put(numpy.asarray(x)), this_ranks_npz['opt_state'].item())



# Create a closure over the optimzation steps:
from utils.optimizer import close_over_optimizer
optimization_fn = close_over_optimizer(args, MPI_AVAILABLE)


from time import perf_counter
from contextlib import contextmanager

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start

times = []

if should_do_io(MPI_AVAILABLE, rank):
    print("Iter.\tTime\tRes")
for i in range(args.iterations):

    with catchtime() as t:
        new_params, proposed_opt_state, residual = optimization_fn(
            dpsi_i, energy, dpsi_i_EL, 
            jacobian,
            # x, spin, isospin,
            w_params, opt_state)
        # tree_map(lambda x : x.block_until_ready(), this_ranks_npz['opt_state'].item())
        residual.block_until_ready()

    if should_do_io(MPI_AVAILABLE, rank):
        print(f"{i}:\t{t():.3f}\t{residual:.4f}", flush=True)
    times.append(t())


if should_do_io(MPI_AVAILABLE, rank):
    times = numpy.asarray(times)
    print(f"First time:  {times[0]:.3f} s.")
    print(f"Mean time after first: {numpy.mean(times[1:]):.3f}s")
