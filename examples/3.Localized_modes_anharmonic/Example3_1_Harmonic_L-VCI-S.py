import VibTools
import vibrations as vib
import numpy as np

print vib.Misc.fancy_box('Example 3:')
print 'Localization of modes of water, harmonic L-VCI-S'
print 'provides initial normal modes\' frequencies and intensities'
print 
# Functions below are used to localize modes in subsets

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
        n = subset[1] - subset[0]
        total += n


    print 'Modes localized: %i, modes in total: %i' %(total, modes.nmodes)

    if total > modes.nmodes:
        raise Exception('Number of modes in the subsets is larger than the total number of modes')
    else:
        cmat = np.zeros((total, total))
        actpos = 0 #actual position in the cmat matrix
        for subset in subsets:
            tmp = localize_subset(modes, range(subset[0], subset[1]))
            modes_mw = np.concatenate((modes_mw, tmp[0]), axis = 0)
            freqs = np.concatenate((freqs, tmp[1]), axis = 0)
            cmat[actpos:actpos + tmp[2].shape[0],actpos:actpos + tmp[2].shape[0]] = tmp[2]
            actpos = actpos + tmp[2].shape[0] 
        localmodes = VibTools.VibModes(total, modes.mol)
        localmodes.set_modes_mw(modes_mw)
        localmodes.set_freqs(freqs)

        return localmodes, cmat


# The vibrations script begins here

# Read in normal modes from SNF results
# using VibTools (LocVib package)

res = VibTools.SNFResults()
res.read()

# Now localize modes in separate subsets

subsets = [[0,3]] 

localmodes,cmat = localize_subsets(res.modes,subsets)

# Define the grid

ngrid = 16
amp = 14
grid = vib.Grid(res.mol,localmodes)
grid.generate_grids(ngrid,amp)

# Generate harmonic 1-mode potentials

v1 = vib.Potential(grid, order=1)
v1.generate_harmonic()

# Generate harmonic 1-mode dipole moments

dmh = vib.Dipole(grid)
dmh.generate_harmonic(res)

# Generate harmonic 2-mode potentials

v2 = vib.Potential(grid, order=2)
v2.generate_harmonic(cmat=cmat)

# Run VSCF calculations for these potentials
# Here we solve only for the vibrational ground state

dVSCF = vib.VSCF2D(v1,v2)
dVSCF.solve()

# Now run VCI calculations using the VSCF wavefunction

VCI = vib.VCI(dVSCF.get_groundstate_wfn(), v1,v2)
VCI.generate_states(1) # singles only
VCI.solve()
VCI.calculate_IR(dmh) # calculate intensities

# Compare the results

# VibTools can provide IR intensities for normal
# and localized modes

irints = res.get_ir_intensity(modes=localmodes)
nirints = res.get_ir_intensity(modes=res.modes)

print 
print
print vib.Misc.fancy_box('Results')
print '%16s %19s %16s' %('Normal','Localized','L-VCI-S')
print '%2s %10s %6s %10s %6s %10s %6s' %('No','Freq.','Int','Freq.','Int','Freq.','Int')
print '-'*56
for i,f in enumerate(localmodes.freqs):
    print '%2i %10.1f %6.1f %10.1f %6.1f %10.1f %6.1f' %(i+1,res.modes.freqs[i],nirints[i],f,irints[i],VCI.energiesrcm[i+1]-VCI.energiesrcm[0],VCI.intensities[i+1])


print 
print 
print vib.Misc.fancy_box('http://www.christophjacob.eu')


