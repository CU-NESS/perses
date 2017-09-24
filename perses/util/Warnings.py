"""

Warnings.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Aug 23 15:26:11 MDT 2015

Description: 

"""
from ares.util import get_hg_rev

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

def ares_version_warning(**kwargs):
    if 'ares_revision' not in kwargs:
        return
    if rank > 0:
        return
    ares_db = kwargs['ares_revision']
    ares_rev = get_hg_rev()
    if ares_db == ares_rev:
        return
    print(("WARNING: database generated with ares revision, {0!s}; BUT, " +\
        "currently using revision, {1!s}.").format(ares_db, ares_rev))

