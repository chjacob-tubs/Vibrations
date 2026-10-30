"""
Pytests (Unittests) for LocVib's Modul: Modes.
"""

import VibTools as vt
import numpy as np
import os
import io
import sys
import pytest


tools_install_path = os.path.dirname(vt.__file__)
data_path = os.path.join(tools_install_path,"tests/test_data")


@pytest.fixture
def H2O_VibToolsMolecule():
    vtmole = vt.VibToolsMolecule()
    vtmole.read_from_coord(os.path.join(data_path,'H2O/coord'))
    return vtmole

@pytest.fixture
def H2O_Mode_data():
    Mode_data = np.load(os.path.join(data_path,'H2O/modes_H2O_test_data.npz'))
    return Mode_data


def test_VibModes_init(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = mol.natoms
    nmodes = 0
    vimo = vt.VibModes(nmodes,mol)
    assert vimo.nmodes == 0
    assert vimo.natoms == 3
    assert len(vimo.freqs) == 0
    assert len(vimo.modes_mw) == 0
    assert len(vimo.modes_c) == 0


def test_set_modes_mw(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    modes_mw = H2O_Mode_data['modes_mw']
    natoms = H2O_Mode_data['natoms']
    # Create Modes class
    vimo = vt.VibModes(3*natoms-6,mol)
    np.testing.assert_equal(vimo.modes_mw, np.zeros((3*natoms-6,3*natoms)))
    # Set massweighted modes
    vimo.set_modes_mw(modes_mw)
    np.testing.assert_equal(vimo.modes_mw, modes_mw)


def test_set_modes_c(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    modes_c = H2O_Mode_data['modes_c']
    natoms = H2O_Mode_data['natoms']
    # Create Modes class
    vimo = vt.VibModes(3*natoms-6,mol)
    np.testing.assert_equal(vimo.modes_c, np.zeros((3*natoms-6,3*natoms)))
    # Set cartesian modes
    vimo.set_modes_c(modes_c)
    np.testing.assert_equal(vimo.modes_c, modes_c)


def test_set_mode_mw(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    # load ref and test data
    ref = H2O_Mode_data
    modes_mw = ref['modes_mw']
    freqs = ref['freqs']
    natoms = mol.natoms
    # Create Modes classi
    vimo = vt.VibModes(3*natoms-6,mol)
    np.testing.assert_equal(vimo.modes_mw, np.zeros((3*natoms-6,3*natoms)))
    # set new single mode in modes
    i = 1
    imode = modes_mw[i,:]
    ifreq = freqs[i]
    vimo.set_mode_mw (i, imode, freq=ifreq)

    np.testing.assert_equal(vimo.modes_mw[0,:],np.zeros(9))
    np.testing.assert_equal(vimo.modes_mw[1,:],imode)
    np.testing.assert_equal(vimo.modes_mw[2,:],np.zeros(9))
    np.testing.assert_equal(vimo.freqs,np.array([1,ifreq,3]))


def test_set_mode_c(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    # load ref and test data
    ref = H2O_Mode_data
    modes_c = ref['modes_c']
    freqs = ref['freqs']
    natoms = mol.natoms
    # Create Modes classi
    vimo = vt.VibModes(3*natoms-6,mol)
    np.testing.assert_equal(vimo.modes_c, np.zeros((3*natoms-6,3*natoms)))
    # set new single mode in modes
    i = 1
    imode = modes_c[i,:]
    ifreq = freqs[i]
    vimo.set_mode_c(i, imode, freq=ifreq)
    np.testing.assert_equal(vimo.modes_c[0,:],np.zeros(9))
    np.testing.assert_equal(vimo.modes_c[1,:],imode)
    np.testing.assert_equal(vimo.modes_c[2,:],np.zeros(9))
    np.testing.assert_equal(vimo.freqs,np.array([1,ifreq,3]))


def test_set_freqs(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    freqs = ref['freqs']
    natoms = mol.natoms
    # Create Modes class
    vimo = vt.VibModes(3*natoms-6,mol)
    np.testing.assert_equal(vimo.freqs, np.array([1,2,3]))
    # set freqs
    vimo.set_freqs(freqs)
    np.testing.assert_equal(vimo.freqs,freqs)


def test_get_modes_c_norm(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref =H2O_Mode_data 
    modes_c = ref['modes_c']
    modes_mw = ref['modes_mw']
    cnorm_ref = ref['modes_cnorm']
    natoms = mol.natoms
    # Create Modes class
    vimo = vt.VibModes(3*natoms-6,mol)
    # Set  modes
    vimo.set_modes_c(modes_c)
    vimo.set_modes_mw(modes_mw)
    # test
    mode_cnorm = vimo.get_modes_c_norm()
    np.testing.assert_equal(mode_cnorm, cnorm_ref)


def test_normalize_modes(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Created according to Bergmanns-Test scheme
    modes.modes_mw = np.zeros((3,9))
    modes.modes_mw[0,1] = 2
    modes.normalize_modes()
    assert modes.modes_mw[0,1] == 1


def test_update_modes_c(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Created according to Bergmanns-Test scheme
    modes.modes_mw = np.zeros((3,9))
    modes.modes_mw[0,0] = 5
    modes.update_modes_c()
    assert pytest.approx(modes.modes_c[0,0],6) == 4.9805514


def test_update_modes_mw(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Created according to Bergmanns-Test scheme
    modes.modes_c = np.zeros((3,9))
    modes.modes_c[0,0] = 5
    modes.update_modes_mw()
    assert pytest.approx(modes.modes_mw[0,0],6) == 5.01952446




def test_get_subset(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_c = ref['modes_c']
    modes_mw = ref['modes_mw']
    freqs = ref['freqs']
    natoms = mol.natoms

    vimo = vt.VibModes(3*natoms-6,mol)
    vimo.set_modes_mw(modes_mw)
    vimo.set_modes_c(modes_c)
    vimo.set_freqs(freqs)

    ml = range(0,2)
    modes = vimo.get_subset(ml)
    np.testing.assert_equal(modes.modes_mw,modes_mw[:2])
    np.testing.assert_equal(modes.modes_c,modes_c[:2])
    np.testing.assert_equal(modes.freqs,freqs[:2])


def test_get_range(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_c = ref['modes_c']
    modes_mw = ref['modes_mw']
    freqs = ref['freqs']
    natoms = mol.natoms

    vimo = vt.VibModes(3*natoms-6,mol)
    vimo.set_modes_mw(modes_mw)
    vimo.set_modes_c(modes_c)
    vimo.set_freqs(freqs)

    modes = vimo.get_range(1000,3700)
    np.testing.assert_equal(modes.modes_mw,modes_mw[:2])
    np.testing.assert_equal(modes.modes_c,modes_c[:2])
    np.testing.assert_equal(modes.freqs,freqs[:2])


def test_write_g98out(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_c = ref['modes_c']
    modes_mw = ref['modes_mw']
    freqs = ref['freqs']
    natoms = mol.natoms

    vimo = vt.VibModes(3*natoms-6,mol)
    vimo.set_modes_mw(modes_mw)
    vimo.set_modes_c(modes_c)
    vimo.set_freqs(freqs)

    vimo.write_g98out()

    import filecmp
    f1 = 'g98.out'
    f2 =os.path.join(data_path, 'H2O/g98_ref.out')

    result = filecmp.cmp(f1, f2, shallow=True)

    if result == False:
        import difflib

        with open(f1) as file_1:
            file_1_text = file_1.readlines()

        with open(f2) as file_2:
            file_2_text = file_2.readlines()

        # Find and print the diff:
        for line in difflib.unified_diff(
                file_1_text, file_2_text, fromfile=f1,
                tofile=f2, lineterm=''):
            print(line)
    assert result == True
    os.remove('g98.out')


def test_get_composition(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Created according to Bergmanns-Test scheme
    g,gn = modes.mol.attype_groups()
    types = modes.get_composition(g)
    ref = [1,1,1]
    i = 0
    for x in gn[:]:
        if x =='H2O':
            for a,b in zip(types[i],ref):
                assert a == pytest.approx(b,5)
        i += 1


def test_print_composition(H2O_VibToolsMolecule):
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    g,gn = modes.mol.attype_groups()
    types = modes.get_composition(g)
    # Act
    try:
        modes.print_composition(gn,types)
        test = True
    except:
        test = False
    # Assert
    assert test == True



def test_print_residue_composition(H2O_VibToolsMolecule):
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Act
    try:
        modes.print_residue_composition()
        test = True
    except:
        test = False
    # Assert
    assert test == True




def test_print_attype_composition(H2O_VibToolsMolecule) :
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Act
    try:
        modes.print_attype_composition()
        test = True
    except:
        test = False
    # Assert
    assert test == True


def test_print_attype2_composition(H2O_VibToolsMolecule) :
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Act
    try:
        modes.print_attype2_composition()
        test = True
    except:
        test = False
    # Assert
    assert test == True


def test_print_attype3_composition(H2O_VibToolsMolecule) :
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Act
    try:
        modes.print_attype3_composition()
        test = True
    except:
        test = False
    # Assert
    assert test == True


def test_print_attype7B_composition(H2O_VibToolsMolecule) :
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Act
    try:
        modes.print_attype7B_composition()
        test = True
    except:
        test = False
    # Assert
    assert test == True


def test_print_attype_all_composition(H2O_VibToolsMolecule):
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Act
    try:
        modes.print_attype_all_composition()
        test = True
    except:
        test = False
    # Assert
    assert test == True


def test_print_atom_composition(H2O_VibToolsMolecule) :
    # Arrange
    mol = H2O_VibToolsMolecule
    natoms = 3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Act
    try:
        modes.print_atom_composition()
        test = True
    except:
        test = False
    # Assert
    assert test == True




def test_transform(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_mw = ref['modes_mw']
    natoms = mol.natoms
    # Create Modes classi
    vimo = vt.VibModes(3*natoms-6,mol)
    vimo_modes_mw = vimo.set_modes_mw(modes_mw)
    # Create unitary transformation matrix 
    tmatrix = np.asarray([[0,0,1],[0,1,0],[1,0,0]])
    transf_modes = vimo.transform(tmatrix)
    transf_modes_mw = transf_modes.modes_mw
    for i in range(3):
        for a,b in zip(transf_modes_mw[i],modes_mw[2-i]):
             np.testing.assert_equal(abs(a),abs(b))


def test_sortmat_by_groups(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = mol.natoms#3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Created according to Bergmanns-Test scheme
    g,gn = modes.mol.attype_groups()
    for c,d in zip(modes.sortmat_by_groups(g),[[0,0,1],[0,1,0],[1,0,0]]):
        for a,b in zip(c,d):
            assert a == b


def test_sortmat_by_freqs(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = mol.natoms#3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Created according to Bergmanns-Test scheme
    for c,d in zip(modes.sortmat_by_freqs(),[[1,0,0],[0,1,0],[0,0,1]]):
        for a,b in zip(c,d):
            assert a == b


def test_sortmat_by_residue(H2O_VibToolsMolecule):
    mol = H2O_VibToolsMolecule
    natoms = mol.natoms#3 # H20
    modes = vt.VibModes(3*natoms-6,mol)
    # Created according to Bergmanns-Test scheme
    for c,d in zip(modes.sortmat_by_residue(),[[0,0,1],[0,1,0],[1,0,0]]):
        for a,b in zip(c,d):
            assert a == b


def test_get_fragment_modes(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_mw = ref['modes_mw']
    modes_frag_ref = ref['get_frag_modes01']
    natoms = mol.natoms
    # Create Modes class
    vimo = vt.VibModes(3*natoms-6,mol)
    vimo_modes_mw = vimo.set_modes_mw(modes_mw)
    ml = [0,1]
    modes = vimo.get_fragment_modes(ml)
    np.testing.assert_equal(modes.modes_mw,modes_frag_ref)


def test_overlap(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_mw = ref['modes_mw']
    natoms = mol.natoms

    vimo_ref = vt.VibModes(3*natoms-6,mol)
    vimo_ref.set_modes_mw(modes_mw)

    ov_ref = ref['ov']

    vimo_test = vt.VibModes(3*natoms-6,mol)
    vimo_test.set_modes_mw(modes_mw)

    ov_unit = np.array([1.,1.,1.])
    ov = vimo_test.overlap(vimo_ref)
    #np.testing.assert_equal(ov,ov_unit)
    np.testing.assert_allclose(ov, ov_unit,atol=4.4408921e-16,rtol=4.4408921e-16)
    for i in range(3):
        vimo_test.modes_mw[i] = vimo_test.modes_mw[i]/(1.5*i+0.7)
    ov = vimo_test.overlap(vimo_ref)
    np.testing.assert_equal(ov,ov_ref)


def test_center(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_mw = ref['modes_mw']
    natoms = mol.natoms


    vimo = vt.VibModes(3*natoms-6,mol)
    vimo.set_modes_mw(modes_mw)

    cen_ref = ref['cen']
    cen0 = vimo.center(0)
    cen1 = vimo.center(1)
    cen2 = vimo.center(2)
    np.testing.assert_equal(cen0,cen_ref[0])
    np.testing.assert_equal(cen1,cen_ref[1])
    np.testing.assert_equal(cen2,cen_ref[2])


def test_distance(H2O_VibToolsMolecule,H2O_Mode_data):
    mol = H2O_VibToolsMolecule
    ref = H2O_Mode_data
    modes_mw = ref['modes_mw']
    natoms = mol.natoms

    vimo = vt.VibModes(3*natoms-6,mol)
    vimo.set_modes_mw(modes_mw)

    dist_ref = ref['dist']

    dist = np.zeros((natoms,natoms))
    for imode in range(natoms):
        for jmode in range(natoms):
            dist[imode,jmode] = vimo.distance(imode, jmode)
    np.testing.assert_equal(dist,dist_ref)


def test_get_tdc_couplingmat():
    # Created according to Bergmanns-Test scheme
    # ! PySNF dependent test !
    # Arrange
    path = os.path.join(data_path,'H2O/')
    snfout_name = 'snf.out'
    restart_name = 'restart'
    coord_name = 'coord'
    res = vt.SNFResults(outname = os.path.join(path,snfout_name),
                    restartname = os.path.join(path,restart_name),
                    coordfile = os.path.join(path,coord_name))
    res.read()

    a = range(0,len(res.modes.freqs))
    modes = res.modes.get_subset(a)

    dipole = res.get_tensor_deriv_nm('dipole',modes=modes)
    # Act Assert
    x = modes.get_tdc_couplingmat(dipole)
    refs = [[ 1.57516798e+03, -1.95910418e+06, -2.82824649e+04],
            [-1.95910418e+06,  3.68532274e+03,  5.85409849e+00],
            [-2.82824649e+04,  5.85409849e+00,  3.79619483e+03]]
    for c,d in zip(x,refs):
        for a,b in zip(c,d):
            assert a == pytest.approx(b, 1)


def test_get_modes_asgn():
    # Created according to Bergmanns-Test scheme
    # ! PySNF dependent test !
    # Arrange
    path = os.path.join(data_path,'H2O/')
    snfout_name = 'snf.out'
    restart_name = 'restart'
    coord_name = 'coord'
    res = vt.SNFResults(outname = os.path.join(path,snfout_name),
                    restartname = os.path.join(path,restart_name),
                    coordfile = os.path.join(path,coord_name))
    res.read()

    a = range(0,len(res.modes.freqs))
    modes = res.modes.get_subset(a)

    g,gn = modes.mol.attype_groups()
    refs = [[171, 172, 173, 174, 175, 176, 177, 178, 179], [180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200], [201, 202, 203, 204, 205, 206, 207, 208, 209, 210], [211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230], [231, 232, 233, 234, 235, 236, 237, 238, 239], [241, 242, 243, 244, 245, 246, 247, 248, 249], [251, 252], [253, 254, 255, 256, 257, 258, 259], [261, 262, 263], [264, 265, 266, 267, 268, 269, 270], [271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290], [292, 293, 294, 295, 296, 297]]
    for c,d in zip(modes.get_modes_asgn(g),refs):
        for a,b in zip(c,d):
            assert a == b


