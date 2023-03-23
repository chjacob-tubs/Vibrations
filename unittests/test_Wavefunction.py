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


####################################
# VIBRATIONS.WAVEFUNCTION UNITTEST #
####################################

def test_Wavefunction_init_empty():
    # Arrange
    # Act
    wf = vib.Wavefunction()
    # Assert
    assert wf.nmodes  == 0 
    assert wf.ngrid   == 0
    assert wf.nstates == 0
    assert len(wf.wfns) == 0


def test_Wavefunction_init(VT_SNFResults_Vib_Grid):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    # Act
    wf = vib.Wavefunction(grids=grid)
    # Assert
    assert wf.nmodes  == 12
    assert wf.ngrid   == 16
    assert wf.nstates == 16
    assert len(wf.wfns) != 0



def test_Wavefunction_init(VT_SNFResults_Vib_Grid):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    # Act
    wf = vib.Wavefunction(grids=grid)
    # Assert
    assert wf.nmodes  == 12
    assert wf.ngrid   == 16
    assert wf.nstates == 16
    assert len(wf.wfns) != 0


def test_Wavefunction_save(VT_SNFResults_Vib_Grid):
    # Arrange
    from time import strftime
    grid = VT_SNFResults_Vib_Grid
    wf = vib.Wavefunction(grids=grid)
    name = 'wavefunctions_' + strftime('%Y%m%d%H%M') + '.npy'  
    # Act
    wf.save_wavefunctions()
    # Assert
    test_wfns = np.load(name)
    np.testing.assert_almost_equal(test_wfns,wf.wfns)
    # Clean up
    os.remove(name)


def test_Wavefunction_read(VT_SNFResults_Vib_Grid):
    # Arrange
    from time import strftime
    grid = VT_SNFResults_Vib_Grid
    wf = vib.Wavefunction(grids=grid)
    ref_wf = 'test_data/ref_wavefunction.npy'
    # Act
    wf_test = vib.Wavefunction()
    wf_test.read_wavefunctions(fname=ref_wf)
    # Assert
    assert len(wf_test.wfns) != 0
    assert wf_test.nmodes  == 12
    assert wf_test.ngrid   == 16
    assert wf_test.nstates == 16


