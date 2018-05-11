import VibTools
import Vibrations as vib
import numpy as np

import os

res = VibTools.SNFResults(outname='snf_h2o/snf.out', 
                          restartname='snf_h2o/restart', 
                          coordfile='snf_h2o/coord')
res.read()

# use the normal modes in the following (no localization)
modes = res.modes

# Define the grid
 
ngrid = 16
amp = 14
grid = vib.Grid(res.mol, modes)
grid.generate_grids(ngrid, amp)

dir_grid = 'grid'
if not os.path.exists(dir_grid):
    os.makedirs(dir_grid)

# write xyz file for equilibrium structure

res.mol.write(os.path.join(dir_grid, 'grid_E0.dat'))

# generate xyz-files for 1-Mode Potentials

for i in range(modes.nmodes):
    for j in range(ngrid):
        print ' *** Writing  Mode: ',i,', Point: ',j
        mol = grid.get_molecule([i],[j])
        filename = 'grid_v1_%i_%i.xyz' % (i,j)
        mol.write(os.path.join(dir_grid, filename))
    print
 
# generate xyz-files for 2-Mode Potentials

for i in range(modes.nmodes):
    for j in range(i+1,modes.nmodes):
        for k in range(ngrid):
            for l in range(ngrid):
                print ' *** Writing V2 - Mode:',i,j,', Point: ',k,l
                mol = grid.get_molecule([i,j],[k,l])
                filename = 'grid_v2_%i_%i_%i_%i.xyz' % (i,j,k,l)
                mol.write(os.path.join(dir_grid, filename))

