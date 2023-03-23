import VibTools
import Vibrations as vib
import numpy as np
import pytest
import os 

## Pre-tests for Grid class
## TEST extended VibTools functions for data production
import sys # added!
sys.path.append("test_data/data_prod/") # added!
import data_prod_funcs_VibTools as datprod
# Prepare Grid input Data:

@pytest.fixture
def VT_SNFResults():
    # VT = VibTools, Vib_Grid = Vibrations.Grid
    # Load SNF-Results via VT
    path = 'test_data/'
    res = VibTools.SNFResults(outname = path+'snf.out',
                              restartname=path+'restart',
                              coordfile=path+'coord')
    res.read()
    return res

@pytest.fixture
def VT_SNFResults_Vib_Grid(VT_SNFResults):
    res = VT_SNFResults
    # Calculate Localmodes via VT - Data-Prod-function:
    subsets = [[0,4], [4,8], [8,12]]
    localmodes,cmat = datprod.localize_subsets(res.modes,subsets)
    # Create Grid Class
    mol = res.mol
    ngrid = 16
    amp = 14
    grid = vib.Grid(mol,localmodes)
    grid.generate_grids(ngrid,amp)
    return grid

@pytest.fixture
def VT_SNFResults_Potentials(VT_SNFResults_Vib_Grid):
    path = 'test_data/'
    grid = VT_SNFResults_Vib_Grid
    # Create 1-mode potentials
    v1 = vib.Potential(grid, order=1)
    # Read in 1-mode potentials
    v1.read_np(path+'1D.npy')
    
    v2 = vib.Potential(grid, order=2)
    v2.read_np(path+'2D.npy')
    return v1, v2


@pytest.fixture
def VSCF_SNFres(VT_SNFResults_Potentials):
    v1, v2 = VT_SNFResults_Potentials
    vscf = vib.VSCF(v1,v2)
    return vscf

@pytest.fixture
def VSCF_Grid_Pots(VT_SNFResults_Vib_Grid,VT_SNFResults_Potentials):
    grid = VT_SNFResults_Vib_Grid
    v1, v2 = VT_SNFResults_Potentials
    vscf = vib.VSCF(v1,v2)
    return vscf, grid, v1,v2

#############################
# VIBRATIONS.VSCF UNITTESTS #
#############################


def test_VSCF_init(VT_SNFResults_Potentials):
    # Arrange
    ref_scf_dx = [10.681926789372667, 10.829512061537628, 10.82966609700452, 11.771463810261395, 
                  8.441139448726034, 8.93333280337, 9.278096996883932, 9.30195862093477, 
                  6.035932992304765, 6.035932992348272, 6.0366972183469585, 6.036697218420436 ]
    v1, v2 = VT_SNFResults_Potentials
    # Act
    vscf = vib.VSCF(v1,v2)
    # Assert
    assert vscf.nmodes       == 12
    assert vscf.ngrid        == 16
    assert vscf.nstates      == 16
    assert vscf.wavefunction != None 
    assert len(vscf.eigv)    != 0
    assert vscf.grids        != None
    np.testing.assert_almost_equal(vscf.dx, ref_scf_dx) 
    assert vscf.solved       == False
    assert vscf.maxpot       == 2


def test_VSCF_collocation(VSCF_Grid_Pots):
    # Arrangei
    vscf, grid, v1,v2 = VSCF_Grid_Pots
    ref_eigval= [0.00118567, 0.00359049, 0.00605896, 0.00858521, 0.01116466, 
                 0.01379397, 0.0164712 , 0.01919598, 0.02197168, 0.02482125,
                 0.02779806, 0.03086768, 0.0338932 , 0.03688944, 0.04185939, 0.04422389]
    ref_phi= [4.92091473e-05, 4.05918153e-04, 2.37124264e-03, 9.98610124e-03, 
              3.08279235e-02, 7.07916322e-02, 1.22361436e-01, 1.60525342e-01, 
              1.60525342e-01, 1.22361436e-01, 7.07916322e-02, 3.08279235e-02, 
              9.98610124e-03, 2.37124264e-03, 4.05918153e-04, 4.92091473e-05]
    grid_input = grid.grids[0]
    v1ind = v1.indices.index(0)
    data_input = v1.data[v1ind]
    # Act
    eigval, phi = vscf._collocation(grid_input,data_input)
    #  Assert
    np.testing.assert_almost_equal(eigval,ref_eigval)
    np.testing.assert_almost_equal(phi[0],ref_phi)


def test_VSCF_norm(VSCF_Grid_Pots):
    # Arrange
    ref_normphi= [4.92091473e-05, 4.05918153e-04, 2.37124264e-03, 9.98610124e-03, 
              3.08279235e-02, 7.07916322e-02, 1.22361436e-01, 1.60525342e-01, 
              1.60525342e-01, 1.22361436e-01, 7.07916322e-02, 3.08279235e-02, 
              9.98610124e-03, 2.37124264e-03, 4.05918153e-04, 4.92091473e-05]
    vscf, grid, v1,v2 = VSCF_Grid_Pots
    grid_input = grid.grids[0]
    v1ind = v1.indices.index(0)
    data_input = v1.data[v1ind]
    eigval, phi = vscf._collocation(grid_input,data_input)
    # Act
    normphi = vscf._norm(phi[0],vscf.dx[0])
    # Assert
    np.testing.assert_almost_equal(normphi,ref_normphi) 


def test_VSCF_get_wave_functions(VSCF_SNFres):
    # Arrange
    ref_wf_00 = np.zeros(16)
    vscf = VSCF_SNFres
    # Act
    wf = vscf.get_wave_functions()
    # Assert
    np.testing.assert_almost_equal(wf[0,0],ref_wf_00)


def test_VSCF_get_wave_function_object(VSCF_SNFres):
    # Arrange
    ref_wf_00 = np.zeros(16)
    vscf = VSCF_SNFres
    # Act
    wf_obj = vscf.get_wave_function_object()
    # Assert
    assert wf_obj.nmodes == 12 
    assert wf_obj.ngrid  == 16
    assert wf_obj.nstates== 16
    np.testing.assert_almost_equal(wf_obj.wfns[0,0],ref_wf_00)


def test_VSCF_save_wave_functions(VSCF_SNFres):
    # Arrange
    vscf = VSCF_SNFres
    from time import strftime
    fname = 'wavefunctions_' + strftime('%Y%m%d%H%M') + '.npy'
    # Act
    vscf.save_wave_functions()
    test = np.load(fname)
    # Assert
    np.testing.assert_almost_equal(test,vscf.wavefunction.wfns)
    # Clean up
    os.remove(fname)

#####################$$$$########
# VIBRATIONS.VSCFDIAG UNITTESTS #
#################################


def test_VSCFDiag_init_1pot(VT_SNFResults_Potentials):
    # Arrange
    v1, v1 = VT_SNFResults_Potentials
    # Act
    vscfDiag = vib.VSCF(v1) 
    # Assert
    assert vscfDiag.nmodes       == 12  
    assert vscfDiag.ngrid        == 16  
    assert vscfDiag.nstates      == 16    
    assert vscfDiag.wavefunction != None
    assert len(vscfDiag.eigv)    != 0
    assert vscfDiag.grids        != None 
    assert len(vscfDiag.dx)      != 0
    assert vscfDiag.solved       == False    
    assert vscfDiag.maxpot       == 1   

def test_VSCFDiag_init_2pot(VT_SNFResults_Potentials):
    # Arrange
    v1, v2 = VT_SNFResults_Potentials
    # Act
    vscfDiag = vib.VSCF(v1,v2)
    # Assert
    assert vscfDiag.nmodes       == 12
    assert vscfDiag.ngrid        == 16
    assert vscfDiag.nstates      == 16
    assert vscfDiag.wavefunction != None
    assert len(vscfDiag.eigv)    != 0
    assert vscfDiag.grids        != None
    assert len(vscfDiag.dx)      != 0
    assert vscfDiag.solved       == False
    assert vscfDiag.maxpot       == 2


def test_VSCFDiag_solve(VT_SNFResults_Potentials):
    # Arrange
    v1, v2 = VT_SNFResults_Potentials
    vscfDiag = vib.VSCFDiag(v1)
    # Act
    vscfDiag.solve()
    # Assert
    assert vscfDiag.nmodes       == 12
    assert vscfDiag.ngrid        == 16
    assert vscfDiag.nstates      == 16
    assert vscfDiag.wavefunction != None
    assert len(vscfDiag.eigv)    == 12
    assert vscfDiag.grids        != None
    assert len(vscfDiag.dx)      == 12
    assert vscfDiag.solved       == True
    assert vscfDiag.maxpot       == 1


def test_VSCFDiag_print_results(VT_SNFResults_Potentials):
    # Arrange
    v1, v2 = VT_SNFResults_Potentials
    vscfDiag = vib.VSCFDiag(v1)
    vscfDiag.solve()
    # Act
    try:
        vscfDiag.print_results()
        test = True
    except:
        test = False
    # Assert
    assert test == True


def test_VSCFDiag_print_eigenvalues(VT_SNFResults_Potentials):
    # Arrange
    v1, v2 = VT_SNFResults_Potentials
    vscfDiag = vib.VSCFDiag(v1)
    vscfDiag.solve()
    # Act
    try:
        vscfDiag.print_eigenvalues()
        test = True
    except:
        test = False
    # Assert
    assert test == True



def test_VSCFDiag_save_wave_functions(VT_SNFResults_Potentials):
    # Arrange
    from time import strftime
    fname ='1D_wavefunctions'
    fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
    v1, v2 = VT_SNFResults_Potentials
    vscfDiag = vib.VSCFDiag(v1)
    vscfDiag.solve()
    # Act
    vscfDiag.save_wave_functions()
    test = np.load(fname)
    # Assert
    np.testing.assert_almost_equal(test,vscfDiag.wavefunction.wfns)
    # Clean up
    os.remove(fname)


###############################
# VIBRATIONS.VSCF2D UNITTESTS #
###############################



def test_VSCF2D_init(VT_SNFResults_Potentials):
    # Arrange
    v1,v2 = VT_SNFResults_Potentials
    # Act
    vscf2D = vib.VSCF2D(v1,v2)
    # Assert
    assert vscf2D.nmodes       == 12
    assert vscf2D.ngrid        == 16
    assert vscf2D.nstates      == 16
    assert vscf2D.wavefunction != None
    assert len(vscf2D.eigv)    == 12
    assert vscf2D.grids        != None
    assert len(vscf2D.dx)      == 12
    assert vscf2D.solved       == False
    assert vscf2D.maxpot       == 1


@pytest.fixture
def VSCF2D_Class(VT_SNFResults_Potentials):
    v1,v2 = VT_SNFResults_Potentials
    vscf2D = vib.VSCF2D(v1,v2)
    return vscf2D


# TODO
# MIT ALTEN TESTSATZ WEITER MACHEN:

def test_VSCF2D_solve(VSCF2D_Class):
    # Arrange
    vscf2D = VSCF2D_Class
    # Act
    vscf2D.solve()
    # Assert
    assert vscf2D.nmodes       == 12
    assert vscf2D.ngrid        == 16
    assert vscf2D.nstates      == 16
    assert vscf2D.wavefunction != None
    assert len(vscf2D.eigv)    == 12
    assert vscf2D.grids        != None
    assert len(vscf2D.dx)      == 12
    assert vscf2D.solved       == True
    assert vscf2D.maxpot       == 1
    assert vscf2D.states       == [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    np.testing.assert_almost_equal(vscf2D.energies, [7274.1364418410985])


def test_VSCF2D_solve_singles(VSCF2D_Class):
    # Arrange
    solved_singles_states = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    energies = [7274.1364418410985, 7790.659594349168, 7825.854758476142, 
                7830.84568521529, 7813.325522270774, 8121.107973117921, 
                8124.462315163319, 8132.626700405709, 8204.66714202618, 
                9431.576797709113, 9447.674260417612, 9463.087841536315, 
                9475.847285669595]
    vscf2D = VSCF2D_Class
    # Act
    vscf2D.solve_singles()
    # Assert
    assert vscf2D.nmodes       == 12
    assert vscf2D.ngrid        == 16
    assert vscf2D.nstates      == 16
    assert vscf2D.wavefunction != None
    assert len(vscf2D.eigv)    == 12
    assert vscf2D.grids        != None
    assert len(vscf2D.dx)      == 12
    assert vscf2D.solved       == True
    assert vscf2D.maxpot       == 1
    assert vscf2D.states       == solved_singles_states
    np.testing.assert_almost_equal(vscf2D.energies,energies)


@pytest.fixture
def VSCF2D_solved_singles(VSCF2D_Class):
    vscf2D = VSCF2D_Class
    vscf2D.solve_singles()
    return vscf2D

def test_VSCF2D__solve_state():
    # is part of VSCF2D.solve() and indirectly tested there
    pass


def test_VSCF2D__veffective():
    # is part of VSCF2D._solve_state() and indirectly tested there
    pass


def test_VSCF2D__scfcorr():
    # is part of VSCF2D._solve_state() and indirectly tested there
    pass



def test_VSCF2D_calculate_intensities():
    #TODO:what are reasonable input data???
    pass





def test_VSCF2D_get_groundstate_wfn(VSCF2D_solved_singles):
    # Arrange
    vscf2DSolvSing = VSCF2D_solved_singles
    # Act
    wfn = vscf2DSolvSing.get_groundstate_wfn()
    # Assert
    assert isinstance(wfn,object) == True




def test_VSCF2D_save_wave_functions(VSCF2D_solved_singles):
# !the method to be tested does not work!
#    # Arrange
#    vscf2DSolvSing = VSCF2D_solved_singles
#    from time import strftime
#    fname = '2D_wavefunctions_' + strftime('%Y%m%d%H%M') + '.npy'
#    # Act
#    vscf2DSolvSing.save_wave_functions(fname)
    pass

def test_VSCF2D_print_results(VSCF2D_solved_singles):
    # Arrange
    vscf2DSolvSing = VSCF2D_solved_singles
    # Act
    try:
        vscf2DSolvSing.print_results()
        test = True
    except:
        test = False
    # Assert
    assert test == True

# TODO:
# TEST: reasonable input test data are missing for the upcoming methods


###############################
# VIBRATIONS.VSCF3D UNITTESTS #
###############################



#def test_VSCF3D_init():
#    pass
#
#def test_VSCF3D_scfcorr():
#    pass
#
#def test_VSCF3D_veffective():
#    pass
#
#
#
################################
## VIBRATIONS.VSCF4D UNITTESTS #
################################
#
#
#def test_VSCF4D_init():
#    pass
#
#def test_VSCF4D_scfcorr():
#    pass
#
#def test_VSCF4D_veffective():
#    pass
#


##-------------------------------------------------------------------
## FURTHER TEST SET: H20 (Data from Tutorial by Julia Brueggemann: JB
##-------------------------------------------------------------------
#
#
#@pytest.fixture
#def JB_H2O_SNFRes():
#    # Read in normal modes from SNF results
#    # using LocVib (LocVib package)
#    path = '/test_data/H2O/'
#    outname     = path+'snf.out'
#    restartname = path+'restart'
#    coordfile   = path+'coord'
#    res = VibTools.SNFResults(outname=outname,restartname=restartname,
#                         coordfile=coordfile)
#    res.read()
#    return res
#
#
#@pytest.fixture
#def JB_H2O_grid(JB_H2O_SNFRes):
#    res = JB_H2O_SNFRes
#    # Now localize modes in separate subsets
#    subsets = [[0],[1,2]]
#    localmodes,cmat = datprod.JB_localize_subsets(res.modes,subsets)
#    # Define the grid
#    ngrid = 16
#    amp = 14
#    grid = vib.Grid(res.mol,localmodes)
#    grid.generate_grids(ngrid,amp)
#    return grid,cmat
#
#
#@pytest.fixture
#def JB_H2O_potentials(JB_H2O_grid):
#    grid,cmat = JB_H2O_grid
#    # Read in anharmonic 1-mode potentials
#    v1 = vib.Potential(grid, order=1)
#    v1.read_np(path+'V1_g16.npy')
#    # Read in anharmonic 2-mode potentials
#    v2h = vib.Potential(grid,order=2)
#    v2h.generate_harmonic(cmat=cmat)
#    return v1, v2h
