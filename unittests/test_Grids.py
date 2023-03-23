import VibTools 
import Vibrations as vib
import numpy as np
import pytest 

# Pre-tests for Grid class
# TEST extended VibTools functions for data production
import sys # added!
sys.path.append("test_data/data_prod/") # added!
import data_prod_funcs_VibTools as datprod

def test_localize_subsets():
    # Arrange 
    path = 'test_data/'
    res = VibTools.SNFResults(outname = path+'snf.out',
                              restartname=path+'restart',
                              coordfile=path+'coord')
    res.read()
    subsets = [[0,4], [4,8], [8,12]]
    locmod_cmat = np.load('test_data/data_prod/input_Grid_data_localmodes.npz')
    #locmod_cmat = np.load('test_data/data_prod/input_Grid_data_localmodes_cmat.npz')
    ref_modes_mw = locmod_cmat['modes_mw']
    ref_freqs = locmod_cmat['freqs']
    ref_cmat = locmod_cmat['cmat']
    # Act 
    localmodes,cmat = datprod.localize_subsets(res.modes,subsets)
    # Assert
    np.testing.assert_almost_equal(localmodes.modes_mw, ref_modes_mw)
    np.testing.assert_almost_equal(localmodes.freqs, ref_freqs)
    np.testing.assert_almost_equal(cmat, ref_cmat)

@pytest.fixture
def VibTools_SNFResults_C2H4():
    path = 'test_data/'
    res = VibTools.SNFResults(outname = path+'snf.out',
                              restartname=path+'restart',
                              coordfile=path+'coord')
    res.read()
    return res

@pytest.fixture
def Vibration_locvibs_C2H4(VibTools_SNFResults_C2H4):
    res = VibTools_SNFResults_C2H4
    subsets = [[0,4], [4,8], [8,12]]
    localmodes,cmat = datprod.localize_subsets(res.modes,subsets)
    return localmodes

@pytest.fixture
def Grid_input_data(VibTools_SNFResults_C2H4,Vibration_locvibs_C2H4):
    res = VibTools_SNFResults_C2H4
    localmodes = Vibration_locvibs_C2H4
    return res, localmodes

@pytest.fixture
def Grid_ref_data():
    Grid_ref = np.load('test_data/data_prod/ref_Grid_data_localmodes.npz')
    return Grid_ref

@pytest.fixture
def Grid_class_setup(Grid_input_data):
    # Arrange
    res, localmodes = Grid_input_data
    mol = res.mol
    ngrid = 16
    amp = 14
    grid = vib.Grid(mol,localmodes)
    grid.generate_grids(ngrid,amp)
    return grid


############################
# VIBRATION.GRID UNITTESTS #
############################


def test_Grid_class_init_empty():
    # Arrange
    # Act
    grid = vib.Grid()
    # Assert
    assert grid.modes  == None 
    assert grid.ngrid  == 0
    assert grid.amp    == 0 
    assert len(grid.grids)  == 0 # empty list []
    assert grid.mol    == None 
    assert grid.natoms == 0
    assert grid.nmodes == 0


def test_Grid_class_init(Grid_input_data):
    # Arrange
    res, localmodes = Grid_input_data
    mol = res.mol
    # Act
    grid = vib.Grid(mol,localmodes)
    # Assert
    assert grid.modes  != None
    assert grid.ngrid  == 0
    assert grid.amp    == 0
    assert len(grid.grids)  == 0 # empty list []
    assert grid.mol    != None
    assert grid.natoms == 6
    assert grid.nmodes == 12


def test_Grid_generate_grids(Grid_input_data, Grid_ref_data):
    # Arrange
    res, localmodes = Grid_input_data
    mol = res.mol
    ngrid = 16
    amp = 14
    grids_ref = Grid_ref_data
    grids_ref = grids_ref['grids']
    # Act
    grid = vib.Grid(mol,localmodes)
    grid.generate_grids(ngrid,amp) 
    # Assert
    assert grid.modes  != None
    assert grid.mol    != None
    assert grid.ngrid  == 16
    assert grid.amp    == 14
    np.testing.assert_almost_equal(grid.grids,grids_ref)
    assert grid.mol    != None
    assert grid.natoms == 6
    assert grid.nmodes == 12


def test_Grid_get_grid_structure(Grid_class_setup, Grid_ref_data):
    # Arrange
    grid = Grid_class_setup
    grids_ref = Grid_ref_data 
    ref_atnums = grids_ref['getgrid_atnums']   
    ref_newcoords = grids_ref['getgrid_newcoords'] 
    list_modes = [0,1]
    points = [10,11]
    # Act
    getgrid_structure = grid.get_grid_structure(list_modes,points)
    # Assert
    getgrid_atnums = getgrid_structure[0]
    getgrid_newcoords = getgrid_structure[1]
    np.testing.assert_almost_equal(getgrid_atnums,ref_atnums)
    np.testing.assert_almost_equal(getgrid_newcoords,ref_newcoords)


def test_Grid_get_molecule(Grid_class_setup, Grid_ref_data):
    # Arrange
    grid = Grid_class_setup
    grids_ref = Grid_ref_data 
    ref_VTmolecule = grids_ref['getVTmolecule']   
    list_modes = [0,1]
    points = [10,11]
    # Act
    molecule = grid.get_molecule(list_modes,points)
    # Assert
    assert molecule.natoms == 6
    np.testing.assert_almost_equal(molecule.coordinates,ref_VTmolecule)



def test_Grid_get_pyadf_molecule(Grid_class_setup, Grid_ref_data):
    # Arrange
    grid = Grid_class_setup
    grids_ref = Grid_ref_data
    ref_Pyadfmol_coord = grids_ref['get_pyadfmol_coords']
    ref_atnums = np.array([1, 6, 6, 1, 1, 1])
    list_modes = [0,1]
    points = [10,11]
    # Act
    PyadfMol = grid.get_pyadf_molecule(list_modes, points)
    # Assert
    np.testing.assert_almost_equal(PyadfMol.get_atomic_numbers(),ref_atnums)
    np.testing.assert_almost_equal(PyadfMol.get_coordinates(),ref_Pyadfmol_coord)


def test_Grid_read_np(Grid_ref_data):
    # Arrange
    grids_ref = Grid_ref_data
    np_grids_ref = grids_ref['np_grids']
    path = 'test_data/'
    # Act
    grid = vib.Grid()
    grid.read_np(path+'grids.npy')
    # Assert
    assert grid.modes  == None
    assert grid.mol    == None
    assert grid.ngrid  == 16
    assert grid.amp    == 0
    np.testing.assert_almost_equal(grid.grids,np_grids_ref)
    assert grid.natoms == 0
    assert grid.nmodes == 12


def test_Grid_get_number_of_grid_points(Grid_class_setup):
    # Arrange
    grid = Grid_class_setup
    # Act
    ngrid = grid.get_number_of_grid_points()
    # Assert
    assert ngrid == 16


def test_Grid_get_number_of_modes(Grid_class_setup):
    # Arrange
    grid = Grid_class_setup
    # Act
    nmodes = grid.get_number_of_modes()
    # Assert
    assert nmodes == 12




def test_Grid_save_grids(Grid_class_setup,Grid_ref_data):
    # Arrange
    import os.path
    grid = Grid_class_setup

    grids_ref = Grid_ref_data
    ref_grid = grids_ref['test_ref_grid']
    # Act
    grid.save_grids(fname='test_save_grids')
    name = 'test_save_grids_12_16.npy'
    test_grid = np.load(name)
    # Assert
    assert os.path.exists(name) == True
    np.testing.assert_almost_equal(test_grid , ref_grid)
    # Clean up
    os.remove(name)



