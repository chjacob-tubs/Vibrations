import VibTools
import Vibrations as vib
import numpy as np

import os

res = VibTools.SNFResults(outname='snf_h2o/snf.out', 
                          restartname='snf_h2o/restart', 
                          coordfile='snf_h2o/coord')
res.read()

# Localize normal modes (in this case, we localize all modes in one subset)
modes = res.modes

modelist = [[0,1,2]]

print('\n\n')
print('*** Localization: ')

localmodes,cmat = VibTools.LocVib.localize_subsets(modelist,res.modes, hessian=True, printing=True, loctype="PM") 

# Define the grid
 
ngrid = 16
amp = 14
grid = vib.Grid(res.mol, localmodes)
grid.generate_grids(ngrid, amp)

# Read in anharmonic 1-mode potentials

v1 = vib.Potential(grid, order=1)
v1.read_np(os.path.join('potentials','V1_g16.npy'))

# Read in anharmonic 1-mode dipole moments

dm1 = vib.Dipole(grid)
dm1.read_np(os.path.join('potentials', 'Dm1_g16.npy'))

# Read in anharmonic 2-mode potentials

v2 = vib.Potential(grid, order=2)
v2.read_np(os.path.join('potentials','V2_g16.npy'))

# Read in anharmonic 2-mode dipole moments

dm2 = vib.Dipole(grid, order=2)
dm2.read_np(os.path.join('potentials','Dm2_g16.npy'))

# Run VSCF calculations for these potentials
# Here we solve only for the vibrational ground state

VSCF = vib.VSCF2D(v1,v2)
VSCF.solve()

# Now run VCI calculations using the VSCF wavefunction

VCI = vib.VCI(VSCF.get_groundstate_wfn(), v1,v2)
VCI.generate_states_nmax(4,4) # VCI-3
VCI.solve()
VCI.calculate_IR(dm1,dm2) # calculate intensities

VCI.print_results(maxfreq=10000,which=3)
VCI.print_contributions(maxfreq=10000,which=2)
