"""
Pytests (Unittests) for LocVib's Modul: PySNF.
"""

#TODO: Provide all saved test data(npz) in one file

import VibTools as vt
import numpy as np
import os
import pytest

tools_install_path = os.path.dirname(vt.__file__)
data_path = os.path.join(tools_install_path,"tests/test_data")


# H2O-Test-Molecule-Setup
@pytest.fixture
def H2O_setup():
    vtmole = vt.VibToolsMolecule()
    vtmole.read_from_coord(os.path.join(data_path,'H2O/coord'))
    return vtmole

# SNFRestartFile Class Tests: =============================================


def test_SNFRestartFile_read_H2O():
    #SNFRestartFile.read(filename)
    filename = os.path.join(data_path,'H2O/restart')
    SNFrest = vt.SNFRestartFile()
    SNFrest.read(filename)
    assert SNFrest.nvar == 9
    assert SNFrest.natoms == 3
    assert SNFrest.iaus == 2
    assert SNFrest.nmodes == 0
    assert SNFrest.displ == 0.01
    assert SNFrest.intonly == False
    assert SNFrest.ncalcsets == 3
    # Load reference data
    filename = os.path.join(data_path,'H2O/H2O_SNFRestartFile_H2O.npz')
    ref_dat = np.load(filename)

    # Numpy Assert test
    np.testing.assert_equal(SNFrest.dipole , ref_dat['dipole'])
    np.testing.assert_equal(SNFrest.gradient , ref_dat['gradient'])
    np.testing.assert_equal(SNFrest.pollen , ref_dat['pollen'])
    np.testing.assert_equal(SNFrest.polvel , ref_dat['polvel'])
    np.testing.assert_equal(SNFrest.gtenlen , ref_dat['gtenlen'])
    np.testing.assert_equal(SNFrest.gtenvel , ref_dat['gtenvel'])
    np.testing.assert_equal(SNFrest.aten , ref_dat['aten'])

    # SNFRestartFile._read_section(f, ncomp) 
    # indirectly tested by SNFRestartFile.read()


def test_SNFRestartFile_del_int_for_atoms_H2O():
    filename = os.path.join(data_path,'H2O/restart')
    SNFrest = vt.SNFRestartFile()
    SNFrest.read(filename)
    atoms = [0,2]
    SNFrest.del_int_for_atoms(atoms)
    zeromat = np.zeros((6,3))
    dipole_all = SNFrest.dipole
    dip_atom0 = dipole_all[0]
    dip_atom1 = dipole_all[2]
    np.testing.assert_equal(dip_atom0 , zeromat)
    np.testing.assert_equal(dip_atom1 , zeromat)


# SNFOutputFile Class Tests: =============================================


def test_SNFOutputFile_init_H2O():
    outfilename = os.path.join(data_path,'H2O/snf.out')
    SNFoutput = vt.SNFOutputFile(outfilename)
    assert SNFoutput.modes ==  None
    assert SNFoutput.lwl ==  None
    assert SNFoutput.filename == os.path.join(data_path,'H2O/snf.out')
    assert SNFoutput.intonly == False

def test_SNFOutputFile_read_H2O(H2O_setup):
    path = os.path.join(data_path,'H2O/')
    outfilename = os.path.join(path,'snf.out')
    SNFoutput = vt.SNFOutputFile(outfilename)
    SNFoutput.read(H2O_setup)
    assert SNFoutput != None  
    assert SNFoutput.filename == os.path.join(path,'snf.out')
    assert SNFoutput.intonly == False
    # Load reference data
    ref_dat_name = os.path.join(path,'H2O_PySNF_modes_freq.npz')
    ref_dat = np.load(ref_dat_name)
    # Test: modes_mw and freqs
    np.testing.assert_equal(SNFoutput.modes.modes_mw, ref_dat['modes_mw'])
    np.testing.assert_equal(SNFoutput.modes.freqs,ref_dat['freqs'])
   

#def test_SNFOutputFile_read_mat():
#    # No reasonable TEST DATA AVAILABLE
#    pass

#def test_SNFOutputFile_read_derivatives():
#    # No reasonable TEST DATA AVAILABLE
#    pass


# SNFControlFile Class Tests: =============================================


#def test_SNFControlFile():
#    # No reasonable TEST DATA AVAILABLE
#    pass


# SNFResultsFile Class Tests: =============================================


def test_SNFResults_init_read_H2O(H2O_setup):
    path = os.path.join(data_path,'H2O/')
    snfout_name = 'snf.out'
    restart_name = 'restart'
    coord_name = 'coord'
    res = vt.SNFResults(outname = os.path.join(path,snfout_name), 
                        restartname = os.path.join(path,restart_name),
                        coordfile = os.path.join(path,coord_name))
    res.read()


    # ----- VibToolsMolecule: mol tests -----

    SNFmol = res.mol
    # Load reference data
    filename = os.path.join(data_path,'H2O/H2O_read_from_coords.npz')
    ref_dat = np.load(filename)

    # Numpy Assert test
    np.testing.assert_equal(SNFmol.natoms, ref_dat['natoms'])
    np.testing.assert_equal(SNFmol.atmasses, ref_dat['atmasses'])
    np.testing.assert_equal(SNFmol.atnums, ref_dat['atnums'])
    np.testing.assert_equal(SNFmol.coordinates, ref_dat['coordinates'])


    # ---- snfoutput tests ------

    SNFoutput = res.snfoutput
    assert SNFoutput != None
    assert SNFoutput.filename == os.path.join(path,'snf.out')
    assert SNFoutput.intonly == False
    # Load reference data
    ref_dat_name = os.path.join(path,'H2O_PySNF_modes_freq.npz')
    ref_dat = np.load(ref_dat_name)
    # Test: modes_mw and freqs
    np.testing.assert_equal(SNFoutput.modes.modes_mw, ref_dat['modes_mw'])
    np.testing.assert_equal(SNFoutput.modes.freqs,ref_dat['freqs'])


    # ---- snfcontrol test ------
    control = res.snfcontrol.modes
    assert control == None


    # ---- snfrestart test ------
    SNFrest = res.restartfile
    filename = os.path.join(data_path,'H2O/restart')
    SNFrest = vt.SNFRestartFile()
    SNFrest.read(filename)
    atoms = [0,2]
    SNFrest.del_int_for_atoms(atoms)
    zeromat = np.zeros((6,3))
    dipole_all = SNFrest.dipole
    dip_atom0 = dipole_all[0]
    dip_atom1 = dipole_all[2]
    np.testing.assert_equal(dip_atom0 , zeromat)
    np.testing.assert_equal(dip_atom1 , zeromat)

    # --- input names / intonly---
    assert res.restartname == os.path.join(path,restart_name)
    assert res.coordfile == os.path.join(path,coord_name)
    assert res.intonly == False


## H2O-Test-Setup
@pytest.fixture
def H2O_SNFsetup():
    path = os.path.join(data_path,'H2O/')
    resPySNF = vt.SNFResults(outname = os.path.join(path,'snf.out'),
                             restartname = os.path.join(path,'restart'), 
                             coordfile = os.path.join(path,'coord'))
    resPySNF.read()    
    a = range(0,len(resPySNF.modes.freqs))
    modes = resPySNF.modes.get_subset(a)
    return resPySNF, modes

## Ala10-Test-Setup
@pytest.fixture
def Ala10_SNFsetup():
    path = os.path.join(data_path,'Ala10/')
    resPySNF = vt.SNFResults(outname = os.path.join(path,'snf2.out'),
                             restartname = os.path.join(path,'restart2'), 
                             coordfile = os.path.join(path,'coord2'))
    resPySNF.read()    
    a = range(0,len(resPySNF.modes.freqs))
    modes = resPySNF.modes.get_subset(a)
    return resPySNF, modes





def test_get_ir_intensity(H2O_SNFsetup):
    # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = H2O_SNFsetup 
   refs = [59.02232365, 2.67880051, 32.56993195]
   # Act + Assert
   for a,b in zip(refs, res.get_ir_intensity(modes)):
       assert a == pytest.approx(b,1e-5)


def test_get_mw_normalmodes(H2O_SNFsetup):
    # Created according to Bergmanns-Test scheme
    # Arrange
    res, modes = H2O_SNFsetup 
    modes1 = modes.modes_mw
    modes2 = res.get_mw_normalmodes()
    # Act + Assert
    for c,d in zip(modes1,modes2):
        for a,b in zip(c,d):
            assert a == pytest.approx(b,1e-05)


def test_get_c_normalmodes(H2O_SNFsetup):
    # Created according to Bergmanns-Test scheme
    # Arrange
    res, modes = H2O_SNFsetup 
    modes1 = modes.modes_c
    modes2 = res.get_c_normalmodes()
    for c,d in zip(modes1,modes2):
        for a,b in zip(c,d):
            assert a == pytest.approx(b,1e-05)


def test_get_tensor_deriv_c(Ala10_SNFsetup):
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup

   tens_list2 = ['dipole','pollen', 'polvel','nvar', 
                 'gtenvel','gtenlen','aten','displ', 
                 'nmodes','natoms', 'iaus', 'n_per_line', 
                 'nvar', 'intonly']

   i = 0
   refs = [0.010994800001706828, -0.2556780000080501, 
          -1.0965000000708685, 309, -0.34466666489548103,
          -0.2376155300377043, 9370.663799995713, 0.01, 0, 103
          , 2, 4, 309, False]
   # Act + Assert
   for ten in tens_list2[:]:
       test2 = eval('res.restartfile.'+ten)
       if type(test2) == int or type(test2) == float or type(test2) == bool:
           np.testing.assert_almost_equal(test2,refs[i])
       else:
           test2 = eval('res.get_tensor_deriv_c(\''+ten+'\')')
           np.testing.assert_almost_equal(test2.sum(),refs[i])
       i += 1


def test_get_tensor_deriv_nm(Ala10_SNFsetup):
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup
   tens_list2 = ['dipole','pollen', 'polvel','nvar', 
                 'gtenvel','gtenlen','aten','displ', 
                 'nmodes','natoms', 'iaus', 'n_per_line', 
                 'nvar', 'intonly']
   i = 0
   refs =[-2.649328307793065, -58.19227495684747, 
          -62.76123656107269, 309, 37.963086509861775, 
          34.747976796126096, 6277.88807883127,0.01, 0, 
          103, 2, 4, 309, False]
   for ten in tens_list2[:]:
       test2 = eval('res.restartfile.'+ten)
       if type(test2) == int or type(test2) == float or type(test2) == bool:
           np.testing.assert_almost_equal(test2,refs[i])
       else:
           test2 = eval('res.get_tensor_deriv_nm(\''+ten+'\')')
           np.testing.assert_almost_equal(test2.sum(),refs[i])
       i += 1


def test_get_tensor_mean(Ala10_SNFsetup):
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup
   
   tens_list2 = ['dipole','pollen', 'polvel','nvar', 
                 'gtenvel','gtenlen','aten','displ', 
                 'nmodes','natoms', 'iaus', 'n_per_line', 
                 'nvar', 'intonly']
   i = 0
   refs =[-4621.966704524, 665943.01434241, 633816.27082699, 
          309, -4807.340965740952, -5044.945419934702, -77429396.747719,
          0.01, 0, 103, 2, 4, 309, False]
   for ten in tens_list2[:]:
       test2 = eval('res.restartfile.'+ten)
       if type(test2) == int or type(test2) == float or type(test2) == bool:
           np.testing.assert_almost_equal(test2,refs[i])
       else:
           test2 = eval('res.get_tensor_mean(\''+ten+'\')')
           np.testing.assert_almost_equal(test2.sum(),refs[i])
       i += 1

# TODO:
#def test_check_consistency(Ala10_SNFsetup):
   # Created according to Bergmanns-Test scheme
   # Arrange
   #res, modes = H2O_SNFsetup
#    print('not tested')

def test_get_a2_invariant(Ala10_SNFsetup) :
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup
   refs = [3.965124569897416e-06, 90.97141390024856]
   mat = res.get_a2_invariant( modes = modes )
   np.testing.assert_almost_equal(refs[0],mat[0],decimal=5)
   np.testing.assert_almost_equal(refs[1],mat.sum(),decimal=5)


def test_get_g2_invariant(Ala10_SNFsetup) :
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup
   refs = [0.04527499563928125, 486.41805589266863]
   mat = res.get_g2_invariant( modes = modes )
   np.testing.assert_almost_equal(refs[0],mat[0],decimal=5)
   np.testing.assert_almost_equal(refs[1],mat.sum(),decimal=5)


def test_get_raman_int(Ala10_SNFsetup) :
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup
   refs = [0.3171034000806141, 7498.640016759866]
   mat = res.get_raman_int(modes = modes )
   np.testing.assert_almost_equal(refs[0],mat[0],decimal=5)
   np.testing.assert_almost_equal(refs[1],mat.sum(),decimal=5)


def test_get_bG_invariant(Ala10_SNFsetup) :
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup 
   refs = [23.599855361006693, -6355.249047300652]
   mat = res.get_bG_invariant( modes = modes )
   np.testing.assert_almost_equal(refs[0],mat[0],decimal=5)
   np.testing.assert_almost_equal(refs[1],mat.sum(),decimal=5)


def test_get_bA_invariant(Ala10_SNFsetup):
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup
   refs = [11.82141118087461, 2092.500764405901]
   mat = res.get_bA_invariant( modes = modes )
   np.testing.assert_almost_equal(refs[0],mat[0],decimal=5)
   np.testing.assert_almost_equal(refs[1],mat.sum(),decimal=5)



def test_get_backscattering_int(Ala10_SNFsetup) :
   # Created according to Bergmanns-Test scheme
   # Arrange
   res, modes = Ala10_SNFsetup
   refs = [0.002252644315896203, -0.8928122953072473]
   mat = res.get_backscattering_int(modes = modes )
   np.testing.assert_almost_equal(refs[0],mat[0],decimal=5)
   np.testing.assert_almost_equal(refs[1],mat.sum(),decimal=5)


