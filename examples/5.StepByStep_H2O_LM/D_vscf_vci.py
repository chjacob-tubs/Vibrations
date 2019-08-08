import VibTools
import Vibrations as vib
import numpy as np

import os

def localize_subset(modes,subset):
    # method that takes normal modes
    # and a range of modes, returns them
    # localized + the cmat
    tmpmodes = modes.get_subset(subset)
    tmploc = VibTools.LocVib(tmpmodes, 'PM')
    tmploc.localize()
    tmploc.sort_by_residue()
    tmploc.adjust_signs()
    tmpcmat = tmploc.get_couplingmat(hessian=True)

    return tmploc.locmodes.modes_mw, tmploc.locmodes.freqs, tmpcmat

def localize_subsets(modes,subsets):
    # method that takes normal modes and list of lists (beginin and end)
    # of subsets and make one set of modes localized in subsets

    # first get number of modes in total
    total = 0
    modes_mw = np.zeros((0, 3*modes.natoms))
    freqs = np.zeros((0,))

    for subset in subsets:
        n = len(subset)
        total += n

    print 'Modes localized: %i, modes in total: %i' %(total, modes.nmodes)

    if total > modes.nmodes:
        raise Exception('Number of modes in the subsets is larger than the total number of modes')
    else:
        cmat = np.zeros((total, total))
        actpos = 0 #actual position in the cmat matrix
        for subset in subsets:
            tmp = localize_subset(modes, subset)
            modes_mw = np.concatenate((modes_mw, tmp[0]), axis = 0)
            freqs = np.concatenate((freqs, tmp[1]), axis = 0)
            cmat[actpos:actpos + tmp[2].shape[0],actpos:actpos + tmp[2].shape[0]] = tmp[2]
            actpos = actpos + tmp[2].shape[0] 
        localmodes = VibTools.VibModes(total, modes.mol)
        localmodes.set_modes_mw(modes_mw)
        localmodes.set_freqs(freqs)

        return localmodes, cmat

########## END OF DEFINITIONS ###########

res = VibTools.SNFResults(outname='snf_h2o/snf.out', 
                          restartname='snf_h2o/restart', 
                          coordfile='snf_h2o/coord')
res.read()

# Localize normal modes (in this case, we localize all modes in one subset)
modes = res.modes

modelist = [[0,1,2]]

print '\n\n'
print '*** Localization: '

localmodes,cmat = localize_subsets(modes,modelist)

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
