import Vibrations as vib
import VibTools
import numpy as np
import pytest
import data_prod_funcs_VibTools as datprod


print('------------- Create grid ------------------\n')
# Load SNF-Results via VT
path = '../'
res = VibTools.SNFResults(outname = path+'snf.out',
                          restartname=path+'restart',
                          coordfile=path+'coord')
res.read()
# Calculate Localmodes via VT - Data-Prod-function:
subsets = [[0,4], [4,8], [8,12]]
localmodes,cmat = datprod.localize_subsets(res.modes,subsets)
# Create Grid Class
mol = res.mol
ngrid = 16
amp = 14
grid = vib.Grid(mol,localmodes)
grid.generate_grids(ngrid,amp)
print("\n\n<=============== Grid is created ==============>\n\n")


print("\n\n ############### Wavefunction  #################\n\n")

print("------ wavefunction init empty -------")

wf = vib.Wavefunction()

print('wf.nmodes  =',   wf.nmodes ) 
print('wf.ngrid   =',   wf.ngrid  )
print('wf.nstates =',   wf.nstates)
print('wf.wfns    =',   wf.wfns   )

print("------ wavefunction init -------")

wf = vib.Wavefunction(grids=grid)

print('wf.nmodes  =',   wf.nmodes ) 
print('wf.ngrid   =',   wf.ngrid  )
print('wf.nstates =',   wf.nstates)
print('wf.wfns    =',   wf.wfns   )

print("------ wavefunction save -------")

wf = vib.Wavefunction(grids=grid)
wf.save_wavefunctions(fname='ref_wavefunctions')

print('wf.nmodes  =',   wf.nmodes ) 
print('wf.ngrid   =',   wf.ngrid  )
print('wf.nstates =',   wf.nstates)
print('wf.wfns    =',   wf.wfns   )

file = np.load('ref_wavefunctions_202211291340.npy')
print(file)
