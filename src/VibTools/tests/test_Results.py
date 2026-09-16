"""Pytest (Unittests) for LocVibs/Vibtools Module: Results."""

import VibTools as vt
import numpy as np
import os
import pytest

tools_install_path = os.path.dirname(vt.__file__)
data_path = os.path.join(tools_install_path,"tests/test_data")

def test_Results_init():
    # Arrange
    # Act
    results = vt.Results()
    # Assert
    assert vt.Molecule.VibToolsMolecule == type(results.mol)
    assert None == results.modes
    assert None == results.lwl

### Abstract class fed with example data:


@pytest.fixture
def H2O_VibToolsMolecule():
    vtmole = vt.VibToolsMolecule()
    vtmole.read_from_coord(os.path.join(data_path,'H2O/coord'))
    return vtmole

@pytest.fixture
def H2O_VibToolsMode(H2O_VibToolsMolecule):
    vtmole = H2O_VibToolsMolecule
    H2O_Mode_data = np.load(os.path.join(data_path,'H2O/modes_H2O_test_data.npz'))
    modes_mw = H2O_Mode_data['modes_mw']
    natoms = H2O_Mode_data['natoms']
    vtmodes = vt.VibModes(3*natoms-6,vtmole)
    return vtmole, vtmodes

@pytest.fixture
def Results_mod(H2O_VibToolsMode):
    vtmole, vtmodes = H2O_VibToolsMode
    results = vt.Results()
    results.mol = vtmole
    results.modes = vtmodes
    return results


def test_Results_read(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.read()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"

def test_Results_get_mw_normalmodes(Results_mod,H2O_VibToolsMode):
    # Arrange
    vtmole, vtmodes = H2O_VibToolsMode
    modes_mw_ref = vtmodes.modes_mw
    Res = Results_mod
    # Act
    modes_mw = Res.get_mw_normalmodes()
    # Assert
    np.testing.assert_almost_equal(modes_mw_ref,modes_mw)


def test_Results_get_c_normalmodes(Results_mod,H2O_VibToolsMode):
    # Arrange
    vtmole, vtmodes = H2O_VibToolsMode
    modes_c_ref = vtmodes.modes_c
    Res = Results_mod
    # Act
    modes_c = Res.get_c_normalmodes()
    # Assert
    np.testing.assert_almost_equal(modes_c_ref,modes_c)




def test_Results_get_tensor_deriv_c(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_tensor_deriv_c(None)
    except Exception as e:
        assert str(e) == "Abstract method not implemented"


######################################################################### 
# Almost all other methods depend on  Results.get_tensor_deriv_c        #
# and the tests do not work because abstract method is not implemented. #
# Only the simple function calls are tested  with "try"                 #
#########################################################################

def test_Results_get_tensor_deriv_nm(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_tensor_deriv_c(None)
    except Exception as e:
        assert str(e) == "Abstract method not implemented"

def test_Results_get_ir_intensity(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_ir_intensity()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"

def test_Results_get_a2_invariant(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_a2_invariant()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"


def test_Results_get_g2_invariant(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_g2_invariant()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"


def test_Results_get_raman_int(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_raman_int()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"


def test_Results_get_aG_invariant(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_aG_invariant()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"


def test_Results_get_bG_invariant(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_bG_invariant()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"



def test_Results_get_bA_invariant(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_bA_invariant()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"


def test_Results_get_backscattering_int(Results_mod):
    # Arrange
    Res = Results_mod
    # Act+Assert
    try:
        Res.get_backscattering_int()
    except Exception as e:
        assert str(e) == "Abstract method not implemented"








