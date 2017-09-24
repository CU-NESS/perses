from perses.beam import BaseBeam, BaseGaussianBeam, BeamUtilities,\
    total_power, polarized

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except:
    rank = 0
    size = 1

try:
    import healpy
except:
    if rank == 0:
        print("WARNING: Healpy not installed.")
