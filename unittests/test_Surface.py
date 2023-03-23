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

def test_Surface_init_empty():
    # Arrange
    # Act
    surf = vib.Surface()
    # Assert
    assert surf.grids  == None   
    assert surf.empty  == True
    assert surf.ngrid  == 0
    assert surf.order  == 1
    assert surf.prop   == (1,)
    assert len(surf.indices)== 0 # empty list check
    assert len(surf.data   )== 0 #    - = -


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


# Prepare Surface class:
@pytest.fixture
def test_Surf_SNFRes(VT_SNFResults_Vib_Grid):
    grids = VT_SNFResults_Vib_Grid
    surf = vib.Surface(grids=grids)
    return surf


#########################
# TESTS with input data #
#########################


def test_Surface_init(VT_SNFResults_Vib_Grid):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    # Act
    surf = vib.Surface(grids=grid)
    # Assert
    assert surf.grids  != None
    assert surf.empty  == False
    assert surf.ngrid  == 16
    assert surf.order  == 1
    assert surf.prop   == (1,)
    assert len(surf.indices)== 0 # empty list check
    assert len(surf.data   )== 0 #    - = -


def test_Surface_delete(test_Surf_SNFRes):
    # Arrange
    surf = test_Surf_SNFRes
    # Act
    surf.delete()
    # Assert
    assert surf.grids  != None
    assert surf.empty  == False
    assert surf.ngrid  == 16
    assert surf.order  == 1
    assert surf.prop   == (1,)
    assert len(surf.indices)== 0 # empty list check
    assert len(surf.data   )== 0 #    - = -

def test_Surface_zero(test_Surf_SNFRes):
    # Arrange
    surf = test_Surf_SNFRes
    # Act
    surf.zero()
    # Assert
    assert surf.grids  != None
    assert surf.empty  == False
    assert surf.ngrid  == 16
    assert surf.order  == 1
    assert surf.prop   == (1,)
    assert len(surf.indices)== 0 # empty list check
    assert len(surf.data   )== 0 #    - = -






#################################
# VIBRATION.POTENTIAL UNITTESTS #
#################################


@pytest.fixture
def potential_grid_input():
    path = 'test_data/'
    grid = vib.Grid()  # Create an empty grid  
    grid.read_np(path+'grids.npy') # Read in an existing grid
    return grid

@pytest.fixture
def potential_1D_2D(potential_grid_input):
    grid = potential_grid_input
    v1 = vib.Potential(grid, order=1)
    v2 = vib.Potential(grid, order=2)
    v1.read_np('test_data/1D.npy')
    v2.read_np('test_data/2D.npy')
    return v1, v2


def test_Potential_class_init_empty():
    # Arrange
    # Act
    pot = vib.Potential()
    # Assert
    assert pot.grids  == None
    assert pot.empty  == True
    assert pot.ngrid  == 0
    assert pot.order  == 1
    assert pot.prop  == (1,)
    assert len(pot.indices) == 0
    assert len(pot.data)    == 0


def test_Potential_class_init_Grid_order1_2(potential_grid_input):
    # Arrange
    grid = potential_grid_input
    # TEST: Order = 1
    # Act
    v1 = vib.Potential(grid, order=1) # Create 1-mode potentials 
    # Assert
    assert v1.grids  != None
    assert v1.empty  == False
    assert v1.ngrid  == 16
    assert v1.order  == 1
    assert v1.prop  == (1,)
    assert len(v1.indices) == 0
    assert len(v1.data)    == 0
    # TEST: Order = 2
    # Act
    v2 = vib.Potential(grid, order=2) # Create 1-mode potentials 
    # Assert
    assert v2.grids  != None
    assert v2.empty  == False
    assert v2.ngrid  == 16
    assert v2.order  == 2
    assert v2.prop  == (1,)
    assert len(v2.indices) == 0
    assert len(v2.data)    == 0


def test_Potential_class_read_np(potential_grid_input,potential_1D_2D):
    ## Arrange
    ref_indices=  [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (7, 8), (7, 9), (7, 10), (7, 11), (8, 9), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11)]
    grid = potential_grid_input
    # Create 1-mode potentials
    v1 = vib.Potential(grid, order=1)
    v2 = vib.Potential(grid, order=2)
    # Act 
    v1.read_np('test_data/1D.npy')
    v2.read_np('test_data/2D.npy')
    # Assert 1D:
    assert v1.grids  != None
    assert v1.empty  == False
    assert v1.ngrid  == 16
    assert v1.order  == 1
    assert v1.prop  == (1,)
    assert v1.indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert len(v1.data) != 0
    # Assert 2D:
    assert v2.grids  != None
    assert v2.empty  == False
    assert v2.ngrid  == 16
    assert v2.order  == 2
    assert v2.prop  == (1,)
    assert v2.indices == ref_indices
    assert len(v2.data) != 0


def test_Potential_save(potential_grid_input):
    # Arrange
    name = 'pot.npy'
    grid = potential_grid_input
    # Create 1-mode potentials
    v1 = vib.Potential(grid, order=1)
    v1.read_np('test_data/1D.npy')
    # Act
    v1.save(fname=name)
    test = np.load(name)
    # Assert
    assert os.path.exists(name) == True
    np.testing.assert_almost_equal(test,v1.data)
    # Clean up
    os.remove(name)



def test_Potential_generate_harmonic(potential_1D_2D):
    # Arrange
    v1,v2 = potential_1D_2D
    file = np.load('test_data/data_prod/ref_Surface_Potential_data.npz')
    cmat = file['cmat']
    v1_ref = file['v1_data']
    v2_ref = file['v2_data']
    # Act
    v1.generate_harmonic(cmat=cmat)
    v2.generate_harmonic(cmat=cmat)
    # Assert
    np.testing.assert_almost_equal(v1.data,v1_ref)



#def test_Potential_generate_from_function():
#   pass





######################################
# VIBRATION.POLARIZABILITY UNITTESTS #
######################################
# TODO: 2D Surface

def test_Polarizability_class_init_empty():
    # Arrange
    # Act
    polar = vib.Polarizability() 
    # Assert
    assert polar.grids  == None
    assert polar.empty  == True
    assert polar.ngrid  == 0
    assert polar.order  == 1
    assert polar.prop   == (6,)
    assert len(polar.indices)== 0 # empty list check
    assert len(polar.data   )== 0 #    - = -
    assert polar.gauge == 'len' 



def test_Polarizability_class_init(VT_SNFResults_Vib_Grid):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    # Act
    polar = vib.Polarizability(grids=grid)
    # Assert
    assert polar.grids  != None
    assert polar.empty  == False
    assert polar.ngrid  == 16
    assert polar.order  == 1
    assert polar.prop   == (6,)
    assert len(polar.indices)== 0 # empty list check
    assert len(polar.data   )== 0 #    - = -
    assert polar.gauge == 'len'


def test_Polarizabilit_generate_harmonic(VT_SNFResults_Vib_Grid,VT_SNFResults):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    polar = vib.Polarizability(grids=grid)
    # Load SNF-Results via VT
    res = VT_SNFResults
    # Act
    polar.generate_harmonic(res)
    # Assert
    np.testing.assert_equal(polar.indices,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])



##############################
# VIBRATION.GTensor UNITTEST #
##############################
#TODO 2D Surface

def test_Gtensor_class_init_empty():
    # Arrange
    # Act
    gtens = vib.Gtensor()
    # Assert
    assert gtens.grids  == None
    assert gtens.empty  == True
    assert gtens.ngrid  == 0
    assert gtens.order  == 1
    assert gtens.prop   == (9,)
    assert len(gtens.indices)== 0 # empty list check
    assert len(gtens.data   )== 0 #    - = -
    assert gtens.gauge == 'vel'


def test_Gtensor_class_init(VT_SNFResults_Vib_Grid):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    # Act
    gtens = vib.Gtensor(grids=grid)
    # Assert
    assert gtens.grids  != None
    assert gtens.empty  == False
    assert gtens.ngrid  == 16
    assert gtens.order  == 1
    assert gtens.prop   == (9,)
    assert len(gtens.indices)== 0 # empty list check
    assert len(gtens.data   )== 0 #    - = -
    assert gtens.gauge == 'vel'


def test_Gtensor_generate_harmonic(VT_SNFResults_Vib_Grid,VT_SNFResults):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    gtens = vib.Gtensor(grids=grid)
    # Load SNF-Results via VT
    res = VT_SNFResults
    # Act
    gtens.generate_harmonic(res)
    # Assert
    np.testing.assert_equal(gtens.indices,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])





##############################
# VIBRATION.Atensor UNITTEST #
##############################


def test_Atensor_class_init_empty():
    # Arrange
    # Act
    atens = vib.Atensor()
    # Assert
    assert atens.grids  == None
    assert atens.empty  == True
    assert atens.ngrid  == 0
    assert atens.order  == 1
    assert atens.prop   == (27,)
    assert len(atens.indices)== 0 # empty list check
    assert len(atens.data   )== 0 #    - = -




def test_Atensor_class_init(VT_SNFResults_Vib_Grid):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    # Act
    atens = vib.Atensor(grids=grid)
    # Assert
    assert atens.grids  != None
    assert atens.empty  == False
    assert atens.ngrid  == 16
    assert atens.order  == 1
    assert atens.prop   == (27,)
    assert len(atens.indices)== 0 # empty list check
    assert len(atens.data   )== 0 #    - = -


def test_Atensor_generate_harmonic(VT_SNFResults_Vib_Grid, VT_SNFResults):
    # Ararnge
    res = VT_SNFResults
    grid = VT_SNFResults_Vib_Grid
    atens = vib.Atensor(grids=grid)
    # Act
    atens.generate_harmonic(res)
    # Assert
    assert atens.grids  != None
    assert atens.empty  == False
    assert atens.ngrid  == 16
    assert atens.order  == 1
    assert atens.prop   == (27,)
    assert len(atens.indices)!= 0 # empty list check
    assert len(atens.data   )!= 0 #    - = -


#############################
# VIBRATION.Dipole UNITTEST #
#############################
# TODO: Order=2 tests

def test_Dipole_class_init_empty():
    # Arrange
    # Act
    dip = vib.Dipole()
    # Assert
    assert dip.grids  == None
    assert dip.empty  == True
    assert dip.ngrid  == 0
    assert dip.order  == 1
    assert dip.prop   == (3,)
    assert len(dip.indices)== 0 # empty list check
    assert len(dip.data   )== 0 #    - = -


def test_Dipole_class_init():
    # Arrange
    # Act
    dip = vib.Dipole()
    # Assert
    assert dip.grids  == None
    assert dip.empty  == True
    assert dip.ngrid  == 0
    assert dip.order  == 1
    assert dip.prop   == (3,)
    assert len(dip.indices)== 0 # empty list check
    assert len(dip.data   )== 0 #    - = -



def test_Dipole_generate_harmonic(VT_SNFResults_Vib_Grid,VT_SNFResults):
    # Arrange
    grid = VT_SNFResults_Vib_Grid
    res = VT_SNFResults
    # Act
    dip = vib.Dipole(grids=grid)
    dip.generate_harmonic(res)
    # Assert
    assert dip.grids  != None
    assert dip.empty  == False
    assert dip.ngrid  == 16
    assert dip.order  == 1
    assert dip.prop   == (3,)
    assert len(dip.indices)!= 0 # empty list check
    assert len(dip.data   )!= 0 #    - = -


def test_Dipole_read_np(VT_SNFResults_Vib_Grid):
    # Arrange
    grid=VT_SNFResults_Vib_Grid
    dip = vib.Dipole(grids=grid)
    # Act
    dip.read_np('test_data/Dm1_g16.npy')
    # Assert
    assert dip.grids  != None
    assert dip.empty  == False
    assert dip.ngrid  == 16
    assert dip.order  == 1
    assert dip.prop   == (3,)
    assert len(dip.data   )!= 0 #    - = -
    np.testing.assert_equal(dip.indices,[0,1,2])





#def test_pinned_Dipole_read_np(pinned,VT_SNFResults_Vib_Grid):
#    # Arrange
#    grid=VT_SNFResults_Vib_Grid
#    dip = vib.Dipole(grids=grid)
#    # Act
#    dip.read_np('test_data/Dm1_g16.npy')
#    # Assert
#    assert isinstance(dip.grids,object) == pinned
#    assert dip.empty  == pinned
#    assert dip.ngrid  == pinned
#    assert dip.order  == pinned
#    assert isinstance(dip.prop,tuple)   == pinned
#    assert isinstance(dip.data,np.ndarray) ==  pinned #    - = -
#    assert dip.indices == pinned
#
#

