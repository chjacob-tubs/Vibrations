"""
Pytests (Unittests) for LocVib's Modul: LocVib.
"""
import VibTools as vt
import numpy as np
import os
import pytest
import copy # Bergmann Test


tools_install_path = os.path.dirname(vt.__file__)
data_path = os.path.join(tools_install_path, "tests/test_data")


@pytest.fixture
def H2O_VibToolsMolecule():
    vtmole = vt.VibToolsMolecule()
    vtmole.read_from_coord(os.path.join(data_path, 'H2O/coord'))
    return vtmole


@pytest.fixture
def H2O_Mode_data():
    Mode_data = np.load(os.path.join(data_path, 'H2O/modes_H2O_test_data.npz'))
    return Mode_data


@pytest.fixture
def H2O_VibToolsMode(H2O_VibToolsMolecule,H2O_Mode_data):
    modes_mw = H2O_Mode_data['modes_mw']
    natoms   = H2O_Mode_data['natoms']
    modes_c  = H2O_Mode_data['modes_c']
    freqs    = H2O_Mode_data['freqs']
    vimo = vt.VibModes(3*natoms-6,H2O_VibToolsMolecule)
    vimo.set_modes_mw(modes_mw)
    vimo.set_modes_c(modes_c)
    vimo.set_freqs(freqs)
    return vimo


@pytest.fixture
def H2O_LocVib_data():
    locvib_data = np.load(os.path.join(data_path, 'H2O/locvib_test_data.npz'))
    return locvib_data



def test_LocVib_init(H2O_VibToolsMode):
    # Arrange
    vimo = H2O_VibToolsMode
    # PM loc crit: Act + Assert
    lvPM = vt.LocVib(vimo)
    assert lvPM.natoms == 3
    assert lvPM.nmodes == 3
    assert lvPM.loctype == 'PM'
    assert lvPM.subsets == None
    transmat = lvPM.transmat
    identity = np.identity(3)
    np.testing.assert_equal(transmat, identity)
    # B loc crit: Act + Assert
    lvPM = vt.LocVib(vimo,loctype='B')
    assert lvPM.natoms == 3
    assert lvPM.nmodes == 3
    assert lvPM.loctype == 'B'
    assert lvPM.subsets == None
    transmat = lvPM.transmat
    identity = np.identity(3)
    np.testing.assert_equal(transmat, identity)

#@pytest.fixture
#Def H2O_LocVib_test(H2O_VibToolsMode):
#    vimo = H2O_VibToolsMode
#    lv = vt.LocVib(vimo)
#    return lv


def test_calc_p(H2O_VibToolsMode,H2O_LocVib_data):
    # Arrange
    vimo = H2O_VibToolsMode
    modes_mw = vimo.modes_mw
    # Act
    lv = vt.LocVib(H2O_VibToolsMode)
    # Assert
    assert lv.calc_p(modes_mw) == H2O_LocVib_data['p']
    

def test_calc_ab(H2O_VibToolsMode,H2O_LocVib_data):
    # Arrange
    vimo = H2O_VibToolsMode
    modes_mw = vimo.modes_mw
    nmodes = vimo.nmodes
    A_ref = H2O_LocVib_data['a']
    B_ref = H2O_LocVib_data['b']

    # Act
    lv = vt.LocVib(H2O_VibToolsMode)

    A = np.zeros((nmodes,nmodes))
    B = np.zeros((nmodes,nmodes))
    for i in range(nmodes):
        for j in range(nmodes):
            a , b = lv.calc_ab(modes_mw, i, j)
            A[i,j] = a
            B[i,j] = b    
    # Assert
    np.testing.assert_equal(A, A_ref)
    np.testing.assert_equal(B, B_ref)
 

def test_rotate(H2O_VibToolsMode,H2O_LocVib_data):
    # Arrange
    vimo = H2O_VibToolsMode
    modes_mw = vimo.modes_mw
    nmodes = vimo.nmodes
    Rall_ref = H2O_LocVib_data['Rall']
    alpha_ref = H2O_LocVib_data['alpha']
    # Act
    lv = vt.LocVib(H2O_VibToolsMode)
    Rall = []
    alpha = np.zeros((nmodes,nmodes))
    for i in range(nmodes):
        for j in range(nmodes):
            R , alph = lv.rotate(modes_mw, i, j)
            Rall.append(R)
            alpha[i,j] = alph
    # Assert
    np.testing.assert_almost_equal(Rall, Rall_ref)
    np.testing.assert_almost_equal(alpha, alpha_ref)


def test_try_localize(H2O_VibToolsMode,H2O_LocVib_data):
    # Arrange
    vimo = H2O_VibToolsMode
    modes_mw = vimo.modes_mw
    nmodes = vimo.nmodes
    transmat_ref = H2O_LocVib_data['transmat']
    del_p_ref = H2O_LocVib_data['del_p']
    # Act
    lv = vt.LocVib(H2O_VibToolsMode)
    transmat,del_p = lv.try_localize()
    # Assert
    np.testing.assert_almost_equal(transmat, transmat_ref)
    np.testing.assert_almost_equal(del_p, del_p_ref)


def test_localize(H2O_VibToolsMode,H2O_LocVib_data):
    # Arrange
    vimo = H2O_VibToolsMode
    transmat_ref = H2O_LocVib_data['transmat']
    # Act
    lv = vt.LocVib(H2O_VibToolsMode)
    lv.localize()
    # Assert
    np.testing.assert_almost_equal(lv.transmat, transmat_ref)


def test_localize_subsets(H2O_VibToolsMode, H2O_LocVib_data):
    # Arrange
    submodes = H2O_LocVib_data['submodes']
    subcmat = H2O_LocVib_data['subcmat']
    vimo = H2O_VibToolsMode
    ml = [[0], [1, 2]] 
    # Act
    localmodes, cmat = vt.LocVib.localize_subsets(ml, vimo)
    # Assert
    np.testing.assert_almost_equal(localmodes.modes_mw, submodes)
    np.testing.assert_almost_equal(cmat, subcmat)


def test_localize_automatic_subsets(H2O_VibToolsMode, H2O_LocVib_data):
    # T. Bergmann test
    # Arrange + Act
    lv = vt.LocVib(H2O_VibToolsMode)
    lv.localize_automatic_subsets(maxerr = 1)
    dif = lv.startmodes.modes_mw - lv.locmodes.modes_mw
    # Assert
    np.testing.assert_almost_equal( dif.sum(), -1.0454343057730062, decimal=9)
    np.testing.assert_almost_equal(lv.locmodes.modes_mw, H2O_LocVib_data['autosubmodes'], decimal=5)


# See VCIS: *See: Panek, Hoeske, Jacob, J. Chem. Phys. 150, 054107 (2019)
#        (doi: 10.1063/1.5083186)
def test_try_localize_vcisdiff(H2O_VibToolsMode):
    # T. Bergmann test
    # Arrange + Act
    lv = vt.LocVib(H2O_VibToolsMode)
    vcis_maxdiff, del_p = lv.try_localize_vcisdiff()
    ref_vm = 0.4825713770028415
    ref_dp = 0.8871338845549157
    np.testing.assert_almost_equal(ref_dp, del_p)
    np.testing.assert_almost_equal(ref_vm, vcis_maxdiff)


def test_set_transmat(H2O_VibToolsMode):
    # T. Bergmann test
    # Arrange + Act
    lv = vt.LocVib(H2O_VibToolsMode)
    tmat = np.asarray([[0,0,1],[0,1,0],[1,0,0]])
    lv.set_transmat(tmat)
    for i in range(3):
        for a,b in zip(lv.locmodes.modes_mw[i],lv.startmodes.modes_mw[2-i]):
            np.testing.assert_almost_equal(a,b)


def test_get_couplingmat(H2O_VibToolsMode):
    # T. Bergmann test
    # Arrange + Act
    lv = vt.LocVib(H2O_VibToolsMode)
    cmat = lv.get_couplingmat()
    refs = [[1575.16798, 0., 0.], [0., 3685.32274, 0.], [0.,0., 3796.19483]]
    for c,d in zip(cmat,refs):
        for a,b in zip(c,d):
            np.testing.assert_almost_equal(a,b,decimal=5)

def test_get_vcismat(H2O_VibToolsMode):
    # T. Bergmann test
    # Arrange + Act
    lv = vt.LocVib(H2O_VibToolsMode)
    vcis_mat = lv.get_vcismat()
    refs = [[1575.16798, 0., 0.], [0., 3685.32274, 0.], [0.,0., 3796.19483]]
    # vcis_mat = cmat?
    for c,d in zip(vcis_mat, refs):
        for a,b in zip(c,d):
            np.testing.assert_almost_equal(a,b,decimal=5)


def test_sort_by_residue(H2O_VibToolsMode):
    # Arrange
    lv = vt.LocVib(H2O_VibToolsMode)
    ref = [[0., 0., 1.],[0., 1., 0.],[1., 0., 0.]]
    # Act
    lv.sort_by_residue()
    tmat = lv.transmat
    # Assert
    np.testing.assert_almost_equal(tmat,ref)    


def test_sort_by_groups(H2O_VibToolsMode):
    # Arrange
    modes = H2O_VibToolsMode
    g,gn = modes.mol.attype_groups()
    lv = vt.LocVib(H2O_VibToolsMode)    
    ref = [[0., 0., 1.],[0., 1., 0.],[1., 0., 0.]]
    # Act
    lv.sort_by_groups(g)
    tmat = lv.transmat
    # Assert
    np.testing.assert_almost_equal(tmat,ref)    


def test_sort_by_residue(H2O_VibToolsMode):
    # Arrange
    lv = vt.LocVib(H2O_VibToolsMode)
    ref = [[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]]
    # Act
    lv.sort_by_freqs()
    tmat = lv.transmat
    # Assert
    np.testing.assert_almost_equal(tmat,ref)    


def test_adjust_signs():
    # T. Bergmann test
    # Arrange + Act
    '''
    No changes (Water)
    '''
    pass


def test_invert_signs(H2O_VibToolsMode):
    # T. Bergmann test
    # Arrange + Act
    lv = vt.LocVib(H2O_VibToolsMode)
    lv.invert_signs(0)
    for i in range(3):
        for a,b in zip(lv.locmodes.modes_mw[i],lv.startmodes.modes_mw[i]):
            if i == 0:
                np.testing.assert_almost_equal(a,-b)
            else:
                np.testing.assert_almost_equal(a,b)

def test_flip_modes(H2O_VibToolsMode):
    # T. Bergmann test
    # Arrange + Act
    lv = vt.LocVib(H2O_VibToolsMode)
    old = copy.copy(lv.locmodes.modes_mw)
    lv.flip_modes(0,2)
    for i in range(3):
        for a,b in zip(lv.locmodes.modes_mw[i],old[2-i]):
            np.testing.assert_almost_equal(a,b)


def test_AutoAssign_init(H2O_VibToolsMode):
    # Arrange
    lv = vt.LocVib(H2O_VibToolsMode)
    ref_errmat = [[  0.        , 250.23493347, 304.81226591],
                  [250.23493347,   0.        ,   0.41683006],
                  [304.81226591,   0.41683006,   0.        ]]
    ref_pmat = [[0.00000000e+00, 8.43065984e-03, 1.11928736e-04],
                [8.43065984e-03, 0.00000000e+00, 8.87076011e-01],
                [1.11928736e-04, 8.87076011e-01, 0.00000000e+00]]
    ref_diffmat = [[   0.     , 2110.15476, 2221.02685],
                   [2110.15476,    0.     ,  110.87209],
                   [2221.02685,  110.87209,    0.     ]]
    # Act
    aa = vt.AutomaticAssignment(lv)
    # Assert
    np.testing.assert_almost_equal(aa.errmat , ref_errmat )
    np.testing.assert_almost_equal(aa.pmat , ref_pmat )
    np.testing.assert_almost_equal(aa.diffmat , ref_diffmat )


def test_AutoAssign_calc_ij(H2O_VibToolsMode):
    # Arrange
    AutoAssgn_ref = np.load(os.path.join(data_path,'H2O/AutoAssign_calc_ij_ref.npz'))
    errmat_ref = AutoAssgn_ref['errmat_s']   
    pmat_ref = AutoAssgn_ref['pmat_s']
    diffmat_ref = AutoAssgn_ref['diffmat_s']

    lv = vt.LocVib(H2O_VibToolsMode)
    aa = vt.AutomaticAssignment(lv)
   
    # Act
    aa.calc_ij(2,2)

    # Assert
    N = int(lv.nmodes)
    # Act
    errmat_s  = []
    pmat_s    = []
    diffmat_s = []
    ind_s = []
    N = 3 # H2O
    mat = np.zeros((N,N))
    for i in range(len(mat)):
        for j in range(len(mat)):
            aa.calc_ij(i,j)
            errmat_s.append( aa.errmat)
            pmat_s.append(aa.pmat)
            diffmat_s.append(aa.diffmat)
    # Assert
    for ind in range(N*3):
            np.testing.assert_almost_equal(errmat_s[ind], errmat_ref[ind])
            np.testing.assert_almost_equal(pmat_s[ind], pmat_ref[ind])
            np.testing.assert_almost_equal(diffmat_s[ind], diffmat_ref[ind])


def test_find_maxp(H2O_VibToolsMode):
    # Arrange
    ref_ind_maxp = (1, 2)
    ref_maxp = 0.8870760109825594
    lv = vt.LocVib(H2O_VibToolsMode)
    aa = vt.AutomaticAssignment(lv)
    maxerr = 10
    maxdiff = 1000
    # Act
    ind_maxp, maxp = aa.find_maxp(maxerr, maxdiff)
    # Assert
    assert ind_maxp == ref_ind_maxp
    assert maxp == pytest.approx(ref_maxp,1e-06)


def test_update_subset(H2O_VibToolsMode):
    # Arrange
    ref = [[0],[1],[2]]
    lv = vt.LocVib(H2O_VibToolsMode)
    aa = vt.AutomaticAssignment(lv)
    # Act +  Assert
    for i in range(3):
        aa.update_subset(i)
        np.testing.assert_almost_equal(aa.subsets,ref)


def test_automatic_subsets(H2O_VibToolsMode):
    # Arrange
    ref1 = [[0],[1],[2]]
    lv = vt.LocVib(H2O_VibToolsMode)
    aa = vt.AutomaticAssignment(lv)

    # Act +  Assert
    subsets = aa.automatic_subsets(0.1)
    np.testing.assert_almost_equal(subsets,ref1)
    subsets = aa.automatic_subsets(0.3)
    np.testing.assert_almost_equal(subsets,ref1)
