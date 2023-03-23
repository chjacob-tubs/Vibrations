#import VibTools
import os, sys
import Vibrations as vib
import VibTools as vt
import numpy as np
import pytest
import pickle
import scipy

def blockPrinting(func):
    #https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value
    return func_wrapper


def test_multichoose():
    # Arrange
    ref = [[[1]],[[2]],[[3]],
           [[0, 1], [1, 0]],
           [[0, 2], [1, 1], [2, 0]],
           [[0, 3], [1, 2], [2, 1], [3, 0]],
           [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
           [[0, 0, 2], [0, 1, 1], [0, 2, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0]],
           [[0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0], [1, 0, 2], [1, 1, 1], 
             [1, 2, 0], [2, 0, 1], [2, 1, 0], [3, 0, 0]]
          ]
    # Act
    test = []
    for n in range(1,4):
        for k in range(1,4):
            test.append(vib.multichoose(n,k))
    # Assert
    np.testing.assert_equal(test,ref)


# Arrange Input-Data for VCI-Class - basend on the modules Grid/Potentials
@pytest.fixture
def Grid():
#    @blockPrinting
#    def innerfunc():
    path = 'test_data/'
    # Create an empty grid
    grid = vib.Grid()
    # Read in an existing grid
    grid.read_np(path+'grids.npy')
    return grid
    #grid = innerfunc()
    return grid


@pytest.fixture
def Potentials(Grid):    
    path = 'test_data/'
    # Create grid
    grid = Grid
    # Create 1-mode potentials
    v1 = vib.Potential(grid, order=1)
    # Read in 1-mode potentials
    v1.read_np(path+'1D.npy')
    v2 = vib.Potential(grid, order=2)
    v2.read_np(path+'2D.npy')
    # Perform VSCF with 2-mode potentials
    @blockPrinting
    def VSCF_func(v1,v2):
        VSCF = vib.VSCF2D(v1,v2)
        VSCF.solve_singles()
        return VSCF,v1,v2
    VSCF, v1,v2 = VSCF_func(v1,v2)
    return VSCF, v1,v2

@pytest.fixture
def SNFResults():
    # Read in normal modes from SNF results
    # using VibTools (LocVib package)
    path = 'test_data/'
    res = vt.SNFResults(outname=path+'snf.out',
                        restartname=path+'restart',
                         coordfile=path+'coord')
    res.read()
    return res


@pytest.fixture
def dipole_1mode(SNFResults,Grid):
    grid = Grid
    res = SNFResults
    # Generate harmonic 1-mode dipole moments
    dmh = vib.Dipole(grid)
    dmh.generate_harmonic(res)
    return dmh



class RefGrid(object):
    # VibTools.Grid
    def __init__(self):
        grids = np.load('test_data/data_prod/VibCI/grid_grids_vci_ref.npy')
        self.modes = None
        self.mol = None
        self.ngrid = 16
        self.amp = 0
        self.grids = grids
        self.natoms = 0
        self.nmodes = 12

def assert_GridClass(refgrid,testgrid):
    assert refgrid.modes  == testgrid.modes 
    assert refgrid.mol    == testgrid.mol   
    assert refgrid.ngrid  == testgrid.ngrid 
    assert refgrid.amp    == testgrid.amp   
    np.testing.assert_almost_equal(refgrid.grids,testgrid.grids)
    assert refgrid.natoms == testgrid.natoms
    assert refgrid.nmodes == testgrid.nmodes

class RefSurfaceDipole(object):
    # VibTools.Surface.Dipole
    def __init__(self):
        # VibTools.Grids
        grids = RefGrid()
        self.grids = grids
        self.empty = False
        self.ngrid = 16
        self.order = 1
        self.prop = (3,)
        self.indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        data = np.load('test_data/data_prod/VibCI/Ref_VT_SurfaceDipole_Class_data.npy')
        self.data = data

def assert_SurfaceDipoleClass(refSD,testSD):
    assert_GridClass(refSD.grids,testSD.grids)
    assert refSD.empty   == testSD.empty   
    assert refSD.ngrid   == testSD.ngrid    
    assert refSD.order   == testSD.order    
    assert refSD.prop    == testSD.prop      
    assert refSD.indices == testSD.indices 
    np.testing.assert_almost_equal(refSD.data,testSD.data)     




@pytest.fixture
def RefDatVCI():
    file = np.load('test_data/data_prod/VibCI/ref_dat_VCI.npz')
    return file


class RefVCI(object):
    def __init__(self):
        file = np.load('test_data/data_prod/VibCI/ref_dat_VCI.npz')
        # Attributes
        self.grids   = file['grids']            
        self.wfns    = file['wfns']
        self.nmodes       = 12
        self.ngrid        = 16
        self.states       = None
        self.combinations = None
        self.solved       = False
        self.dx           = file['dx']
        self.a            = file['a']
        self.coefficients = file['coefficients']
        self.sij          = file['sij'] 
        self.tij          = file['tij'] 
        self.store_ints   = True
        self.integrals    = {}
        self.store_potints = True
        self.int1d       = {}  
        self.int2d       = {}   
        self.int3d       = {}   
        self.int4d       = {}  
        self.fortran     = False  
        self.energies    = []
        self.energiesrcm = [] 
        self.H           = np.array([])
        self.vectors     = np.array([]) 
        self.intensities = None
        self.maxpot      = 2
        self.v1_indices = file['v1_indices']
        self.v1_data    = file['v1_data']   
        self.v2_indices = file['v2_indices']
        self.v2_data    = file['v2_data']   


@pytest.fixture
def Ref_VCI():
    refVCI = RefVCI()
    return refVCI


@pytest.fixture
def Ref_solved_VCI():
    # Create standard class
    refsolvedVCI = RefVCI()
    # update to solved vci class
    filename = 'test_data/data_prod/VibCI/solved_vib_module_VCI_ref_dat.npz'
    file = np.load(filename,allow_pickle=True)
    refsolvedVCI.states      = file['states']
    integrals                = file['integrals']
    refsolvedVCI.integrals   = integrals.item()
    refsolvedVCI.int1d       = file['int1d'].item()
    refsolvedVCI.int2d       = file['int2d'].item()
    refsolvedVCI.energies    = file['energies']
    refsolvedVCI.energiesrcm = file['energiesrcm']
    refsolvedVCI.H           = file['H']
    refsolvedVCI.vectors     = file['vectors']
    refsolvedVCI.solved      = True
    refsolvedVCI.nmax        = 1
    refsolvedVCI.smax        = 1
    return refsolvedVCI



def assert_vci_class(refvci,vci):
    # converting classes to dictionaries
    vci = vci.__dict__
    refvci = refvci.__dict__
    # Create keyword list
    keys_test = list(vci.keys())
    keys = list(refvci.keys())
    ## check keys
    for i in range(len(keys)):
        key = keys[i]
        vci_attr = vci[key]
        refvci_attr = refvci[key]
        if type(vci_attr) == np.ndarray  or  type(refvci_attr) == np.ndarray:
            if key == 'wfns' or key == 'coefficients':
                # sign reversal possible
                vci_attr = abs(vci_attr)
                refvci_attr = abs(refvci_attr)
            assert np.allclose(vci_attr,refvci_attr) == True
        else:
            assert vci_attr == refvci_attr



def assert_solved_vci_class(refvci,vci):
    # converting classes to dictionaries
    vci = vci.__dict__
    refvci = refvci.__dict__
    # Create keyword list
    keys_test = list(vci.keys())
    keys = list(refvci.keys())
    ## check keys
    for i in range(len(keys)):
        key = keys[i]
        vci_attr = vci[key]
        refvci_attr = refvci[key]
#        print('==============================')
        print('key:',key)     
        print('type(vci_attr)=',type(vci_attr))
        print('type(refvci_attr)=',type(refvci_attr))
        if type(vci_attr) == scipy.sparse.csr.csr_matrix:
            vci_attr = np.array(vci_attr)
        elif type(vci_attr) == np.ndarray  or  type(refvci_attr) == np.ndarray:
            if key == 'wfns' or key == 'coefficients':
                # sign reversal possible
                vci_attr = abs(vci_attr)
                refvci_attr = abs(refvci_attr)
            if key == 'vectors':
                vci_attr = abs(vci_attr)
                refvci_attr = abs(refvci_attr)        
            assert np.allclose(vci_attr,refvci_attr) == True
        elif type(vci_attr) == dict and type(refvci_attr) == dict:
            dickey = list(vci_attr.keys())
            ref_dickey = list(refvci_attr.keys())
            if dickey == []:
                vci_attr == refvci_attr
 #               print('langweilig!')
            else:
                assert np.allclose(dickey,ref_dickey) == True          
          #      print('vci_attr.keys',dickey)
          #      print('dickey[0]',dickey[0])
          #      print('dickey[0]',dickey[0])
          #      print('vci_attr[dickey[0]]',vci_attr[dickey[0]])
                for i in range(len(dickey)):
                    k = dickey[i]
                #    print('k=',k)
                    dic = vci_attr[k]
                #    print('dic=',dic)
                    ref_dic = refvci_attr[k]
                #    print('ref_dic=',ref_dic)
                    assert np.allclose(abs(dic),abs(ref_dic)) == True 
        else:
            assert vci_attr == refvci_attr


@pytest.fixture
def vciCalcIR(Potentials,dipole_1mode):
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    dmh = dipole_1mode
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    vci.calculate_IR(dmh)
    return vci







def test_VCI_init(Potentials,Ref_VCI):
    # Arrange
    refvci = Ref_VCI
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    # Act 
    vci = vib.VCI(gwfn, v1,v2)
    # Assert
    assert_vci_class(refvci,vci)






def test_calculate_transition():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_transition():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_diagonal():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_single():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_double():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_quadriple():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_triple():
    pass 
    # Arrange
    # Act
    # Assert



def test_order_of_transition():
    pass 
    # Arrange
    # Act
    # Assert



def test_nex_state():
    pass 
    # Arrange
    # Act
    # Assert



def test_nexmax_state():
    pass 
    # Arrange
    # Act
    # Assert



def test_idx_fundamentals():
    pass 
    # Arrange
    # Act
    # Assert


#TODO: Print

def test_print_results(vciCalcIR):
    # Arrange
    vci = vciCalcIR
    # Act
    try:
        vci.print_results()
        test = True
    except:
        test = False
    # Assert
    assert test == True

def test_print_short_state(vciCalcIR):
    # Arrange
    vci = vciCalcIR
    # Act
    try:
        vci.print_results()
        test = True
    except:
        test = False
    # Assert
    assert test == True




def test_print_contributions(vciCalcIR):
    # Arrange
    vci = vciCalcIR
    # Act
    try:
        vci.print_contributions()
        test = True
    except:
        test = False
    # Assert
    assert test == True




def test_print_states(vciCalcIR):
    # Arrange
    vci = vciCalcIR
    # Act
    try:
        vci.print_states()
        test = True
    except:
        test = False
    # Assert
    assert test == True




def test_generate_states_nmax(Potentials,Ref_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    ## Ref
    refvci = Ref_VCI
    refvci.states = [[0,0,0,0,0,0,0,0,0,0,0,0], 
                     [0,0,0,0,0,0,0,0,0,0,0,1], 
                     [0,0,0,0,0,0,0,0,0,0,1,0], 
                     [0,0,0,0,0,0,0,0,0,1,0,0], 
                     [0,0,0,0,0,0,0,0,1,0,0,0], 
                     [0,0,0,0,0,0,0,1,0,0,0,0], 
                     [0,0,0,0,0,0,1,0,0,0,0,0], 
                     [0,0,0,0,0,1,0,0,0,0,0,0], 
                     [0,0,0,0,1,0,0,0,0,0,0,0], 
                     [0,0,0,1,0,0,0,0,0,0,0,0], 
                     [0,0,1,0,0,0,0,0,0,0,0,0], 
                     [0,1,0,0,0,0,0,0,0,0,0,0], 
                     [1,0,0,0,0,0,0,0,0,0,0,0]]
    # Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states_nmax(1,1)
    # Assert 
    assert_vci_class(refvci,vci)
    assert vci.nmax == 1
    assert vci.smax == 1


def test_generate_states(Potentials,Ref_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    ## Ref
    refvci = Ref_VCI
    refvci.states = [[0,0,0,0,0,0,0,0,0,0,0,0], 
                     [0,0,0,0,0,0,0,0,0,0,0,1], 
                     [0,0,0,0,0,0,0,0,0,0,1,0], 
                     [0,0,0,0,0,0,0,0,0,1,0,0], 
                     [0,0,0,0,0,0,0,0,1,0,0,0], 
                     [0,0,0,0,0,0,0,1,0,0,0,0], 
                     [0,0,0,0,0,0,1,0,0,0,0,0], 
                     [0,0,0,0,0,1,0,0,0,0,0,0], 
                     [0,0,0,0,1,0,0,0,0,0,0,0], 
                     [0,0,0,1,0,0,0,0,0,0,0,0], 
                     [0,0,1,0,0,0,0,0,0,0,0,0], 
                     [0,1,0,0,0,0,0,0,0,0,0,0], 
                     [1,0,0,0,0,0,0,0,0,0,0,0]]
    # Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    # Assert 
    assert_vci_class(refvci,vci)
    assert vci.nmax == 1
    assert vci.smax == 1


def test_filter_states():
    pass 
    # Arrange
    # Act
    # Assert



def test_filter_combinations():
    pass 
    # Arrange
    # Act
    # Assert



def test_combgenerator():
    pass 
    # Arrange
    # Act
    # Assert



def test_combgenerator_nofilter():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_transition_moments():
    pass 
    # Arrange
    # Act
    # Assert


    

#@pytest.fixture
#def generate_dm1_dm2():
#    import pickle
#    path = 'test_data/DAR/'
#    
#    # get the VCI class that produced reference data
#    # from openbabel import openbabel
#    import VibTools as LocVib
#    from numpy import linalg as LA
#    # from matplotlib import pyplot
#    import pickle
#    import os
#
#    def localize_subset(modes,subset):
#        # method that takes normal modes
#        # and a range of modes, returns them
#        # localized + the cmat
#        tmpmodes = modes.get_subset(subset)
#        tmploc = LocVib.LocVib(tmpmodes, 'PM')
#        tmploc.localize()
#        tmploc.sort_by_residue()
#        tmploc.adjust_signs()
#        tmpcmat = tmploc.get_couplingmat(hessian=True)
#
#        return tmploc.locmodes.modes_mw, tmploc.locmodes.freqs, tmpcmat
#
#    def localize_subsets(modes,subsets):
#        # method that takes normal modes and list of lists (beginin and end)
#        # of subsets and make one set of modes localized in subsets
#
#        # first get number of modes in total
#        total = 0
#        modes_mw = np.zeros((0, 3*modes.natoms))
#        freqs = np.zeros((0,))
#
#        for subset in subsets:
#            n = len(subset)
#            total += n
#
#
#        print('Modes localized: %i, modes in total: %i' %(total, modes.nmodes))
#
#        if total > modes.nmodes:
#            raise Exception('Number of modes in the subsets is larger than the total number of modes')
#        else:
#            cmat = np.zeros((total, total))
#            actpos = 0 #actual position in the cmat matrix
#            for subset in subsets:
#                tmp = localize_subset(modes, subset)
#                modes_mw = np.concatenate((modes_mw, tmp[0]), axis = 0)
#                freqs = np.concatenate((freqs, tmp[1]), axis = 0)
#                cmat[actpos:actpos + tmp[2].shape[0],actpos:actpos + tmp[2].shape[0]] = tmp[2]
#                actpos = actpos + tmp[2].shape[0] 
#            localmodes = LocVib.VibModes(total, modes.mol)
#            localmodes.set_modes_mw(modes_mw)
#            localmodes.set_freqs(freqs)
#
#            return localmodes, cmat
#
#
#    # path = os.path.join(os.getcwd(),'anharm')
#
#    res = LocVib.SNFResults(outname=os.path.join(path,'snf.out'),
#                            restartname=os.path.join(path,'restart'),
#                            coordfile=os.path.join(path,'coord'))
#    res.read()
#
#    subsets = np.load(os.path.join(path,'subsets.npy'))
#    localmodes,cmat = localize_subsets(res.modes,subsets)
#
#    ngrid = 16
#    amp = 14
#    grid = vib.Grid(res.mol,localmodes)
#    grid.generate_grids(ngrid,amp)
#
#    dm1 = vib.Dipole(grid)
#    dm1.read_np(os.path.join(path,'Dm1_g16.npy'))
#
#    dm2 = vib.Dipole(grid, order=2)
#    dm2.read_np(os.path.join(path,'Dm2_g16.npy'))
#    
#    return dm1, dm2
#    
#
#def test_calculate_transition_matrix(generate_dm1_dm2):
#    # Arrange
#    path = 'test_data/DAR/'
#    ref_transm = np.load(path+'VCI_dipolemoments.npy')
#    ref_inten = np.load(path+'VCI_frequencies.npy')
#    ref_freqs = np.load(path+'VCI_intensities.npy')
#    
#    # Act
#    dm1, dm2 = generate_dm1_dm2
#    
#    fileObj = open(path+'VCI.obj', 'rb')
#    VCI2 = pickle.load(fileObj) 
#    fileObj.close()
#
#    transm, inten, freqs = VCI2.calculate_transition_matrix([dm1],[dm2])
#    
#    # Assert
#    np.testing.assert_almost_equal(transm, ref_transm) 
#    np.testing.assert_almost_equal(inten,  ref_inten) 
#    np.testing.assert_almost_equal(freqs,  ref_freqs) 



def test_calculate_IR(Potentials,dipole_1mode,Ref_solved_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    # Input for act:
    dmh = dipole_1mode
    ## Ref - VCI - Class
    refvci = Ref_solved_VCI
    refvci.intensities = np.array([0.00000000e+00, 1.55678662e+00, 1.55964169e-13, 1.00410688e+02, 
                         5.23256215e-05, 5.08841050e-16, 2.72826737e-16, 1.58504326e+01, 
                         1.56996935e-16, 6.94398781e-16, 1.87393864e+01, 7.55607489e-18, 
                         2.17779853e+01])

    ref_dm1 = RefSurfaceDipole() # VibTools.Surface.Dipole class
    # Pre-Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    # Act
    vci.calculate_IR(dmh)
    # Assert
    assert_solved_vci_class(refvci,vci)
    dm1_dict = vci.dm1.__dict__
    print('dm1_keys: ', list(dm1_dict.keys()))
    assert_SurfaceDipoleClass(ref_dm1,vci.dm1)
    assert vci.dm2 == None
    #TODO: assert vci.prop1[0] == dm1 ?  
    assert_SurfaceDipoleClass(ref_dm1,vci.prop1[0])

    assert vci.prop2 == None
    assert vci.transitions == None 





def test_calculate_intensities():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_raman():
    pass 
    # Arrange
    # Act
    # Assert



def test_calculate_roa():
    pass 
    # Arrange
    # Act
    # Assert



def test__v1_integral(): 
    pass 
    # Arrange
    # Act
    # Assert



def test__v2_integral():
    pass 
    # Arrange
    # Act
    # Assert



def test_v2_integral():
    pass 
    # Arrange
    # Act
    # Assert



def test__v3_integral():
    pass 
    # Arrange
    # Act
    # Assert



def test__v4_integral():
    pass 
    # Arrange
    # Act
    # Assert



def test__ovrlp_integral(): 
    pass 
    # Arrange
    # Act
    # Assert



def test__kinetic_integral(): 
    pass 
    # Arrange
    # Act
    # Assert



def test__dgs_kinetic_integral(): 
    pass 
    # Arrange
    # Act
    # Assert



def test__dgs_ovrlp_integral(): 
    pass 
    # Arrange
    # Act
    # Assert



def test__chi(): 
    pass 
    # Arrange
    # Act
    # Assert



def test__calculate_coeff():
    pass 
    # Arrange
    # Act
    # Assert



def test__calculate_kinetic_integrals():
    pass 
    # Arrange
    # Act
    # Assert



def test__calculate_ovrlp_integrals():
    pass 
    # Arrange
    # Act
    # Assert


def test_solve(Potentials,Ref_solved_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    ## Ref
    refvci = Ref_solved_VCI

    # (Pre-)Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    # Act
    vci.solve()
    # Assert 
    assert_solved_vci_class(refvci,vci)



def test_save_vectors():
    pass 
    # Arrange
    # Act
    # Assert



def test_read_vectors():
    pass 
    # Arrange
    # Act
    # Assert
