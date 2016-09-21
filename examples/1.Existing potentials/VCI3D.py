import vibrations as vib

print vib.Misc.fancy_box('Example 1:')
print 'Use of existing grids and potentials.'
print 
print '  Data taken from: http://pes-database.theochem.uni-stuttgart.de/surfaces/index.php'
print '  By Guntram Rauhut and co-workers'

# Create an empty grid
grid = vib.Grid()

# Read in an existing grid
grid.read_np('grids.npy')

# Create 1-mode potentials
v1 = vib.Potential(grid, order=1)
# Read in 1-mode potentials
v1.read_np('1D.npy')

v2 = vib.Potential(grid, order=2)
v2.read_np('2D.npy')

# Perform VSCF with 2-mode potentials
VSCF = vib.VSCF2D(v1,v2)
VSCF.solve_singles()

# VCI part

# Initialize a VCI object with the VSCF groundstate wavefunction and the 
# potentials
VCI = vib.VCI(VSCF.get_groundstate_wfn(), v1,v2)

# Define the excitation space, how many modes with how many excitation 
# quanta
VCI.generate_states_nmax(1,1)

# Solve the VCI
VCI.solve()

print 
print 
print vib.Misc.fancy_box('http://www.christophjacob.eu')
