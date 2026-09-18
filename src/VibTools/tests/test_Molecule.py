"""
Pytests (Unittests) for LocVib/VibTools's Module: Molecule.
"""

import VibTools as vt
import numpy as np
import os 
import pytest

#TODO: Provide all saved test data(npz) in one file


tools_install_path = os.path.dirname(vt.__file__)
data_path = os.path.join(tools_install_path,"tests/test_data")


# Test of basic method
def test_read_from_coord():
    ## Arrange
    vtmole = vt.VibToolsMolecule()
    # Load reference data
    filename = os.path.join(data_path,'H2O/H2O_read_from_coords.npz')
    ref_dat = np.load(filename)
    ## Act
    # Create module data
    molefilename = os.path.join(data_path,'H2O/coord')
    vtmole.read_from_coord(molefilename)
    ### Assert 
    np.testing.assert_equal(vtmole.natoms, ref_dat['natoms'])
    np.testing.assert_equal(vtmole.atmasses, ref_dat['atmasses'])
    np.testing.assert_equal(vtmole.atnums, ref_dat['atnums'])
    np.testing.assert_equal(vtmole.coordinates, ref_dat['coordinates'])


# H2O-Test-Setup
@pytest.fixture
def H2O_setup():
    vtmole = vt.VibToolsMolecule()
    vtmole.read_from_coord(os.path.join(data_path,'H2O/coord'))
    return vtmole

# Ala-Test-Setup
@pytest.fixture
def Ala10_setup(): # For Bergmann-Tests
    vtmole = vt.VibToolsMolecule()
    vtmole.read_from_coord(os.path.join(data_path,'Ala10/coord'))
    return vtmole


##################
# Methods tests: #
##################

def test_get_fragment(H2O_setup):
    # Arrange
    atomlist1 = [0,1,2]
    atomlist2 = [1]
    # Act
    frag1 = H2O_setup.get_fragment(atomlist1)
    frag2 = H2O_setup.get_fragment(atomlist2)
    # Assert
    np.testing.assert_almost_equal(frag1.atnums,[1.,8.,1.])
    np.testing.assert_almost_equal(frag2.atnums,[8.])


def test_reset_molcache(H2O_setup):
    H2O_setup._atnums = 1
    H2O_setup.reset_molcache()

    assert H2O_setup._atnums == None
    assert H2O_setup._atmasses == None
    assert H2O_setup._coords == None


def test_write_and_read(H2O_setup):
    # Test write functionality
    H2O_setup = vt.VibToolsMolecule()
    H2O_setup.read_from_coord(os.path.join(data_path,'H2O/coord'))
    filename = 'write_test_temp'
    H2O_setup.write(filename)
    # check existance of file
    assert (True == os.path.exists(filename)), 'Test-file is not created.'

    # test read functionality
    vtmole2 = vt.VibToolsMolecule()
    vtmole2.read(filename)

    # check: Read file == read_from_coord 
    #(with tolerance: 5th decimal place)
    np.testing.assert_equal(H2O_setup.natoms, vtmole2.natoms)
    np.testing.assert_equal(H2O_setup.atmasses, vtmole2.atmasses)
    np.testing.assert_equal(H2O_setup.atnums, vtmole2.atnums)
    np.testing.assert_allclose(H2O_setup.coordinates, vtmole2.coordinates,atol=0.0018,rtol=4e-06)
    # remove test-file
    os.remove('write_test_temp')
    assert(False == os.path.exists(filename)),'Test-file still exists.'


def test_get_natoms(H2O_setup):
    assert H2O_setup.get_natoms() == H2O_setup.natoms


def test_get_atmasses(H2O_setup):
    np.testing.assert_equal(H2O_setup.get_atmasses(), H2O_setup.atmasses)

#TODO
def test_get_atnums(H2O_setup):
    pass


def test_get_coordinates(H2O_setup):
    np.testing.assert_equal(H2O_setup.get_coordinates(), H2O_setup.coordinates)#,atol=0.0018,rtol=4e-06)

def test_add_atoms(H2O_setup):
    # Load reference data
    filename = os.path.join(data_path,'H2O/H2O_add_atom1_111.npz')
    ref_dat111 = np.load(filename) 
    H2O_setup.add_atoms([1],[[-1, 1, 1]])
    np.testing.assert_equal(H2O_setup.atnums, ref_dat111['atnums'])
    np.testing.assert_equal(H2O_setup.coordinates,ref_dat111['coordinates'])


##########################
# Ala10 tests --- groups #
##########################


def test_residue_groups(Ala10_setup):
    # Bergmann-Test
    vtmole = Ala10_setup
    ref1= [[46, 47, 48, 49, 50, 51, 68, 69, 70, 71, 102], [52, 53, 54, 55, 56, 57, 72, 73, 74, 75], [58, 59, 60, 61, 62, 63, 76, 77, 78, 79], [40, 41, 43, 44, 80, 81, 82, 83, 84, 85], [64, 65, 66, 67, 86, 87, 88, 89, 90, 91], [18, 19, 20, 21, 92, 93, 94, 95, 96, 97], [22, 23, 36, 37, 42, 45, 98, 99, 100, 101], [2, 3, 4, 5, 12, 13, 14, 15, 38, 39], [16, 17, 24, 25, 26, 27, 32, 33, 34, 35], [0, 1, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31]]
    ref2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    groups, groupnames = vtmole.residue_groups()
    for c,d in zip(groups,ref1):
        for a,b in zip(c,d):
            assert a == b

    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b



def test_attype_groups(Ala10_setup):
    # Bergmann-Test
    vtmole = Ala10_setup
    ref1= [[8, 12, 32, 48, 54, 60, 82, 88, 94, 100], [1, 11, 15, 35, 45, 57, 63, 85, 91, 97], [7, 17, 39, 47, 53, 59, 81, 87, 93, 99], [13, 30, 33, 49, 55, 61, 83, 89, 95, 101], [6, 16, 38, 46, 52, 58, 80, 86, 92, 98], [10, 14, 34, 42, 50, 56, 62, 84, 90, 96], [0, 2, 18, 22, 24, 40, 64, 68, 72, 76], [3, 4, 5, 19, 20, 21, 23, 25, 26, 27, 28, 29, 31, 36, 37, 41, 43, 44, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79], [9], [51, 102], []]
    ref2 = ['N', 'H', 'C', 'O', 'CA', 'HA', 'CB', 'HB', 'OXT', 'HXT', 'H2O']
    groups, groupnames = vtmole.attype_groups()
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b


def test_attype_groups_2(Ala10_setup):
    # Bergmann-Test
    vtmole = Ala10_setup
    ref1 = [[1, 8, 11, 12, 15, 32, 35, 45, 48, 51, 54, 57, 60, 63, 82, 85, 88, 91, 94, 97, 100, 102], [7, 9, 13, 17, 30, 33, 39, 47, 49, 53, 55, 59, 61, 81, 83, 87, 89, 93, 95, 99, 101], [6, 16, 38, 46, 52, 58, 80, 86, 92, 98], [10, 14, 34, 42, 50, 56, 62, 84, 90, 96], [0, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 36, 37, 40, 41, 43, 44, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]]
    ref2 = ['NH', 'CO', 'CA', 'HA', 'CHB']
    groups, groupnames = vtmole.attype_groups_2()
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b


def test_attype_groups_3(Ala10_setup):
    # Bergmann-Test
    vtmole = Ala10_setup

    ref1= [[1, 8, 11, 12, 15, 32, 35, 45, 48, 51, 54, 57, 60, 63, 82, 85, 88, 91, 94, 97, 100, 102], [7, 9, 13, 17, 30, 33, 39, 47, 49, 53, 55, 59, 61, 81, 83, 87, 89, 93, 95, 99, 101], [6, 16, 38, 46, 52, 58, 80, 86, 92, 98], [0, 2, 18, 22, 24, 40, 64, 68, 72, 76], [10, 14, 34, 42, 50, 56, 62, 84, 90, 96], [5, 21, 27, 31, 37, 44, 67, 71, 75, 79], [3, 19, 23, 25, 28, 41, 65, 69, 73, 77], [4, 20, 26, 29, 36, 43, 66, 70, 74, 78], [], []]
    ref2= ['NH', 'CO', 'CA', 'CB', 'HA', 'HB3', 'CHB1', 'CHB2', 'CHG1', 'CHD1']
    groups, groupnames = vtmole.attype_groups_3()
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b




def test_attype_groups_7B(Ala10_setup):
    # Bergmann-Test
    vtmole = Ala10_setup
    ref1 = [[1, 8, 11, 12, 15, 32, 35, 45, 48, 51, 54, 57, 60, 63, 82, 85, 88, 91, 94, 97, 100, 102], [7, 9, 13, 17, 30, 33, 39, 47, 49, 53, 55, 59, 61, 81, 83, 87, 89, 93, 95, 99, 101], [6, 10, 14, 16, 34, 38, 42, 46, 50, 52, 56, 58, 62, 80, 84, 86, 90, 92, 96, 98], [0, 2, 3, 18, 19, 22, 23, 24, 25, 28, 40, 41, 64, 65, 68, 69, 72, 73, 76, 77], [4, 20, 26, 29, 36, 43, 66, 70, 74, 78], [], [], [], [], [5, 21, 27, 31, 37, 44, 67, 71, 75, 79]]
    ref2= ['NH', 'CO', 'CHA', 'CHB1', 'CHB2', 'CHG1', 'CHD1', 'Ring', 'Term', 'HB3']
    groups, groupnames = vtmole.attype_groups_7B()
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b


def test_attype_all_groups(Ala10_setup):
    # Bergmann-Test
    vtmole = Ala10_setup
    ref1= [[0, 2, 18, 22, 24, 40, 64, 68, 72, 76], [1, 11, 15, 35, 45, 57, 63, 85, 91, 97], [3, 4, 5, 19, 20, 21, 23, 25, 26, 27, 28, 29, 31, 36, 37, 41, 43, 44, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79], [6, 16, 38, 46, 52, 58, 80, 86, 92, 98], [7, 17, 39, 47, 53, 59, 81, 87, 93, 99], [8, 12, 32, 48, 54, 60, 82, 88, 94, 100], [9], [10, 14, 34, 42, 50, 56, 62, 84, 90, 96], [13, 30, 33, 49, 55, 61, 83, 89, 95, 101], [51, 102]]
    ref2= ['CB', 'H', 'HB', 'CA', 'C', 'N', 'OXT', 'HA', 'O', 'HXT']
    groups, groupnames = vtmole.attype_all_groups()
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b




def test_atom_groups(Ala10_setup):
    # Bergmann-Test
    vtmole = Ala10_setup
    ref1 = [[0, 2, 6, 7, 16, 17, 18, 22, 24, 38, 39, 40, 46, 47, 52, 53, 58, 59, 64, 68, 72, 76, 80, 81, 86, 87, 92, 93, 98, 99], [1, 3, 4, 5, 10, 11, 14, 15, 19, 20, 21, 23, 25, 26, 27, 28, 29, 31, 34, 35, 36, 37, 41, 42, 43, 44, 45, 50, 51, 56, 57, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 84, 85, 90, 91, 96, 97, 102], [8, 12, 32, 48, 54, 60, 82, 88, 94, 100], [9, 13, 30, 33, 49, 55, 61, 83, 89, 95, 101]]
    ref2 = ['C', 'H', 'N', 'O']
    groups, groupnames = vtmole.atom_groups()
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b


def test_amide1_groups(Ala10_setup):
    # Created according to Bergmanns-Test scheme
    vtmole = Ala10_setup
    res = 4
    ref1= [[81, 83], [88, 91], [94, 97], [45, 100], [86, 90], [80, 84], [92, 96], [7, 13, 17, 30, 33, 39, 47, 49, 53, 55, 59, 61, 87, 89, 93, 95, 99, 101], [0, 2, 18, 22, 24, 40, 64, 68, 72, 76], [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 19, 20, 21, 23, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 41, 42, 43, 44, 46, 48, 50, 51, 52, 54, 56, 57, 58, 60, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 82, 85, 98, 102]]
    ref2= ['CO', 'NH', 'NH2', 'NH3', 'CAH', 'CA0', 'CA2', 'COo', 'CBa', 'R']

    groups, groupnames = vtmole.amide1_groups(res)
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b


def test_amide2_groups(Ala10_setup):
    # Created according to Bergmanns-Test scheme
    vtmole = Ala10_setup
    res = 4
    ref1= [[59, 61], [82, 85], [81, 83], [88, 91], [87, 89], [94, 97], [40, 41, 43, 44, 64, 65, 66, 67, 80, 84, 86, 90], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 92, 93, 95, 96, 98, 99, 100, 101, 102]]
    ref2 = ['CO-1', 'NH', 'CO', 'NH+1', 'CO+1', 'NH+2', 'CH01', 'R']

    groups, groupnames = vtmole.amide2_groups(res)
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b

def test_amide3_groups(Ala10_setup):
    # Created according to Bergmanns-Test scheme
    vtmole = Ala10_setup
    res = 4
    ref1 = [[59, 61, 82, 85], [80, 84], [81, 83, 88, 91], [86, 90], [87, 89, 94, 97], [92, 96], [18, 19, 20, 21, 40, 41, 43, 44, 64, 65, 66, 67], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 93, 95, 98, 99, 100, 101, 102]]
    ref2 = ['Am-1/0', 'CA0', 'Am0/1', 'CA+1', 'Am1/2', 'CA+2', 'CB', 'R']

    groups, groupnames = vtmole.amide3_groups(res)
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b


def test_skelCN_groups(Ala10_setup):
    # Created according to Bergmanns-Test scheme
    vtmole = Ala10_setup
    res = 4
    ref1 = [[80, 82], [84], [81, 83], [91], [86, 88], [90], [64, 65, 66, 67], [87, 89], [97], [92, 94], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 85, 93, 95, 96, 98, 99, 100, 101, 102]]
    ref2= ['NCA0', 'HA0', 'CO0', 'H+1', 'NCA+1', 'HA+1', 'CHB+1', 'CO+1', 'H+2', 'NCA+2', 'R']
    groups, groupnames = vtmole.skelCN_groups(res)
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b


def test_ch3_groups(Ala10_setup):
    # Created according to Bergmanns-Test scheme
    vtmole = Ala10_setup
    res = 4
    ref1 = [[81, 83], [88, 91], [86, 90], [64, 65, 66, 67], [87, 89], [94, 97], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 85, 92, 93, 95, 96, 98, 99, 100, 101, 102]]
    ref2 = ['CO0', 'NH+1', 'CA+1', 'CB+1', 'CO+1', 'NH+2', 'R']
    groups, groupnames = vtmole.ch3_groups(res)
    for c,d in zip(ref1,groups):
        for a,b in zip(c,d):
            assert a == b
    for c,d in zip(groupnames,ref2):
        for a,b in zip(c,d):
            assert a == b

