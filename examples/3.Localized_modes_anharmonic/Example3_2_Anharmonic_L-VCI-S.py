import VibTools
import Vibrations as vib
import numpy as np

print(vib.Misc.fancy_box('Example 3:'))
print('Localization of modes of water, harmonic L-VCI-S')
print('provides initial normal modes\' frequencies and intensities')
print()

# Read in normal modes from SNF results
# using VibTools (LocVib package)

res = VibTools.SNFResults()
res.read()

# Now localize modes in separate subsets

subsets = [list(range(0,3))]

localmodes,cmat = VibTools.LocVib.localize_subsets(subsets,res.modes,hessian=True,printing=True,loctype="PM")

# Define the grid

ngrid = 16
amp = 14
grid = vib.Grid(res.mol,localmodes)
grid.generate_grids(ngrid,amp)

# Read in anharmonic 1-mode potentials

v1 = vib.Potential(grid, order=1)
v1.read_np('V1_g16.npy')

# Read in anharmonic 1-mode dipole moments

dm1 = vib.Dipole(grid)
dm1.read_np('Dm1_g16.npy')

# Read in anharmonic 2-mode potentials

v2 = vib.Potential(grid, order=2)
v2.read_np('V2_g16.npy')

# Read in anharmonic 2-mode dipole moments

dm2 = vib.Dipole(grid, order=2)
dm2.read_np('Dm2_g16.npy')

# Run VSCF calculations for these potentials
# Here we solve only for the vibrational ground state

dVSCF = vib.VSCF2D(v1,v2)

dVSCF.solve()

# Now run VCI calculations using the VSCF wavefunction

VCI = vib.VCI(dVSCF.get_groundstate_wfn(), v1,v2)
VCI.generate_states(1) # singles only
VCI.solve()
VCI.calculate_IR(dm1,dm2) # calculate intensities

# Compare the results

# VibTools can provide IR intensities for normal
# and localized modes

irints = res.get_ir_intensity(modes=localmodes)
nirints = res.get_ir_intensity(modes=res.modes)

print()
print()
print(vib.Misc.fancy_box('Results'))
print('%16s %19s %16s' %('Normal','Localized','L-VCI-S'))
print('%2s %10s %6s %10s %6s %10s %6s' %('No','Freq.','Int','Freq.','Int','Freq.','Int'))
print('-'*56)
for i,f in enumerate(localmodes.freqs):
    print('%2i %10.1f %6.1f %10.1f %6.1f %10.1f %6.1f' %(i+1,res.modes.freqs[i],nirints[i],f,irints[i],VCI.energiesrcm[i+1]-VCI.energiesrcm[0],VCI.intensities[i+1]))


print()
print()
print(vib.Misc.fancy_box('http://www.christophjacob.eu'))


