import VibTools 
import os, sys
import Vibrations as vib
import VibTools as vt
import numpy as np
import pytest
import pickle
import scipy


#TODO: Tests are created by trial and error.
#      Summarize redundant test arrangements into pytest.fixtures

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



def assert_VibTools_Molecule(refVTMolecule,testVTMolecule):
    # VT = VibTools; VT.Molecule
    print("Assert VibTools.Molecule Class")
    np.testing.assert_almost_equal(refVTMolecule.natoms,testVTMolecule.natoms)
    np.testing.assert_almost_equal(refVTMolecule.atmasses, testVTMolecule.atmasses)
    np.testing.assert_almost_equal(refVTMolecule.atnums     ,testVTMolecule.atnums)
    np.testing.assert_almost_equal(refVTMolecule.coordinates,testVTMolecule.coordinates)


def assert_VibTools_Modes(refVTModes, testVTModes):
    print("Assert VibTools.Modes Class")
    assert refVTModes.nmodes   == testVTModes.nmodes  
#    np.testing.assert_almost_equal( refVTModes.mol      ,  testVTModes.mol    ) 
    if refVTModes.mol != None: 
        assert_VibTools_Molecule(refVTModes.mol,testVTModes.mol)
    np.testing.assert_almost_equal( refVTModes.natoms   ,  testVTModes.natoms  )
    np.testing.assert_almost_equal( refVTModes.freqs    ,  testVTModes.freqs   )
    np.testing.assert_almost_equal( refVTModes.modes_mw ,  testVTModes.modes_mw)
    np.testing.assert_almost_equal( refVTModes.modes_c  ,  testVTModes.modes_c )


def assert_GridClass(refgrid,testgrid):
    print("Assert GridClass:")    
    if refgrid.mol == None:
        assert testgrid.mol == None
    else:
        assert_VibTools_Molecule(refgrid.mol,testgrid.mol)  

    if refgrid.modes == None:
        assert testgrid.modes == None
    else:
        assert_VibTools_Modes(refgrid.modes,testgrid.modes) 

    
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
    print("Assert SurfaceDipoleClass")
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
        print('\n------ key:',key)
        vci_attr = vci[key]
        refvci_attr = refvci[key]
        if type(vci_attr) == np.ndarray  or  type(refvci_attr) == np.ndarray:
            if key == 'wfns' or key == 'coefficients':
                # sign reversal possible
                vci_attr = abs(vci_attr)
                refvci_attr = abs(refvci_attr)
   #         assert np.allclose(vci_attr,refvci_attr) == True
                np.testing.assert_almost_equal( vci_attr,refvci_attr)
        else:
            assert vci_attr == refvci_attr



def assert_solved_vci_class(refvci,vci,show=False):
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
        if show == True:
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
#            assert np.allclose(vci_attr,refvci_attr) == True
            np.testing.assert_almost_equal(vci_attr,refvci_attr)

        elif type(vci_attr) == dict and type(refvci_attr) == dict:
            dickey = list(vci_attr.keys())
            ref_dickey = list(refvci_attr.keys())
            if dickey == []:
                vci_attr == refvci_attr
            else:
#                assert np.allclose(dickey,ref_dickey) == True          
                np.testing.assert_almost_equal(dickey,ref_dickey)          
                for i in range(len(dickey)):
                    k = dickey[i]
                    dic = vci_attr[k]
                    ref_dic = refvci_attr[k]
#                    assert np.allclose(abs(dic),abs(ref_dic)) == True 
                    np.testing.assert_almost_equal(abs(dic),abs(ref_dic))  
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






def test_calculate_transition(DAR_Set, DAR_comb_generator_c):
    # arrange 
    ## input
    vci2, dm1, dm2 = DAR_Set
    # this input provides same output as vci(dar).combgenerator() = c
    c_list = DAR_comb_generator_c
    ## reference
    refvci2 = RefVCI2_DAR_solved()
    ref_triple_list= [(0, 0, 0.004778267246682157), (0, 1, -1.235878924578447e-07), 
                      (0, 2, 9.668211647974323e-07), (1, 1, 0.014260336755916174), 
                      (1, 2, -1.070959739485084e-06), (2, 2, 0.023648145874827876)]
    # act
    triple_list = []
    for i in range(len(c_list)):
        triple = vci2.calculate_transition(c_list[i])
        triple_list.append(triple)    
    # assert
#    assert ref_triple_list == triple_list 
    np.testing.assert_almost_equal(ref_triple_list,triple_list) 

    ## assert attributes
    assert_solved_vci_class(refvci2,vci2)



def test_calculate_diagonal(DAR_Set, DAR_comb_generator_c):
    # arrange 
    ## input
    vci2, dm1, dm2 = DAR_Set
    # this input provides same output as vci(dar).combgenerator() = c
    c_list = DAR_comb_generator_c
    ## reference
    refvci2 = RefVCI2_DAR_solved() 
    ref_tmp_list= [0.004778267246682157, -1.235878924578447e-07, 
                   9.668211647974323e-07, 0.014260336755916174, 
                   -1.070959739485084e-06, 0.023648145874827876]         
    # act
    tmp_list = []
    for i in range(len(c_list)):
        tmp = vci2.calculate_diagonal(c_list[i])
        tmp_list.append(tmp)    
    # assert
#    assert ref_tmp_list == tmp_list 
    np.testing.assert_almost_equal( ref_tmp_list,tmp_list) 
    ## assert attributes
    assert_solved_vci_class(refvci2,vci2)



def test_calculate_single(DAR_Set, DAR_comb_generator_c):
    # arrange 
    ## input
    vci2, dm1, dm2 = DAR_Set
    # this input provides same output as vci(dar).combgenerator() = c
    c_list = DAR_comb_generator_c
    index = [1,2,4]
    ## reference
    refvci2 = RefVCI2_DAR_solved()
    ref_tmp_list=[-1.235878924578447e-07, 
                   9.668211647974323e-07, 
                  -1.070959739485084e-06] 
    # act
    tmp_list = []
    for i in range(len(index)):
        tmp = vci2.calculate_single(c_list[index[i]])
        tmp_list.append(tmp)
        print(tmp)
    # assert
    np.testing.assert_almost_equal( ref_tmp_list,tmp_list)
    ## assert attributes
    assert_solved_vci_class(refvci2,vci2)






def test_calculate_double(Potentials, dipole_1mode,Ref_solved_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    # Input for act:
    dmh = dipole_1mode
    ## Ref - VCI - Class
    refvci = Ref_solved_VCI
    # Pre-Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    # Input c: combgenerator
    c= (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1, 12)
    reftmp= 2.10252938124636e-05
    # Act
    tmp = vci.calculate_double(c)
#    # Assert
    np.testing.assert_almost_equal(abs(tmp),abs(reftmp))
    assert_solved_vci_class(refvci,vci)


#TODO: Reference data needed
#def test_calculate_quadriple():
#    pass 
    # Arrange
    # Act
    # Assert
#def test_calculate_triple():
#    pass 
    # Arrange
    # Act
    # Assert



def test_order_of_transition(DAR_Set, DAR_comb_generator_c):
    # Arrange 
    ## Input
    VCI2, dm1, dm2 = DAR_Set
    # This input provides same output as VCI(DAR).combgenerator() = c
    c_list = DAR_comb_generator_c
    ## Reference
    refVCI2 = RefVCI2_DAR_solved() 
    ref_order = [0,1,1,0,1,0]
    # Act
    order = []
    for i in range(len(c_list)):
        count = VCI2.order_of_transition(c_list[i])
        order.append(count)
    # Assert
    assert ref_order == order


def test_nex_state(Ref_solved_VCI,Potentials):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    # Input for act:
    dmh = dipole_1mode
    ## Ref - VCI - Class
    refvci = Ref_solved_VCI
    # Pre-Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    refcontri = [np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), 
                 np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])]
    # Act
    contri = vci.nex_state(vci.states)
    # Assert
    np.testing.assert_almost_equal(refcontri,contri)
    assert_solved_vci_class(refvci,vci)


def test_nexmax_state(Potentials,Ref_solved_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    # Input for act:
    dmh = dipole_1mode
    ## Ref - VCI - Class
    refvci = Ref_solved_VCI
    # Pre-Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    refcontri = (1,np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # Act
    contri = vci.nexmax_state(vci.states)
    # Assert
    np.testing.assert_almost_equal(refcontri[:1],contri[:1])
    assert_solved_vci_class(refvci,vci)



def test_idx_fundamentals(Potentials,Ref_solved_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    # Input for act:
    dmh = dipole_1mode
    ## Ref - VCI - Class
    refvci = Ref_solved_VCI
    # Pre-Act
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    refidx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]    
    # Act
    idx = vci.idx_fundamentals()
    # Assert
    np.testing.assert_almost_equal(refidx,idx)
    assert_solved_vci_class(refvci,vci)

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
    ref = '0: '
    # Act
    s = vci.print_short_state(vci.states[0])
    # Assert
    assert s == ref



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


def test_filter_states(Potentials,Ref_solved_VCI):
    # Arrange
    filename = 'test_data/data_prod/VibCI/Ref_input_VCI_filter_states.npz'
    file = np.load(filename)
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    states = file['states_input']
    nexc = 1
    func = lambda x :len([_f for _f in x if _f]) < nexc + 1
    vci.states = states
    ## Ref
    refvci = Ref_solved_VCI
    ref_states = file['states_ref']
    refvci.states = ref_states
    # Act
    vci.filter_states(func)
    # Assert
    np.testing.assert_almost_equal(ref_states, vci.states)
    #assert_vci_class(refvci,vci)
    assert_solved_vci_class(refvci,vci)



#TODO: Check if the vci attribute is necessary!  self.combinations
def test_filter_combinations(Potentials,Ref_solved_VCI):
    # Arrange
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    ## Ref
    refvci = Ref_solved_VCI
    # Act
    vci.filter_combinations()
    # Assert
    assert_solved_vci_class(refvci,vci)








def test_combgenerator(DAR_Set,DAR_comb_generator_c):
    # Arrange
    ## Ref
    ref_c_list = DAR_comb_generator_c
    refVCI2 = RefVCI2_DAR_solved()
    ## Input
    VCI2, dm1, dm2 = DAR_Set
    # Act # combgenerator = yield function
    c_list = []
    for c in VCI2.combgenerator():
        c_list.append(c)
        print(c)
    # Assert
    assert c_list == ref_c_list
    ## Assert Attributes
    assert_solved_vci_class(refVCI2,VCI2)


def test_combgenerator_nofilter(DAR_Set,DAR_comb_generator_c):
    # This input provides same output as VCI(DAR).combgenerator()
    # Arrange 
    ## Ref
    ref_c_list = DAR_comb_generator_c
    refVCI2 = RefVCI2_DAR_solved()
    ## Input
    VCI2, dm1, dm2 = DAR_Set
    # Act # combgenerator = yield function
    c_list = []
    for c in VCI2.combgenerator_nofilter():
        c_list.append(c)
        print(c)
    # Assert
    assert c_list == ref_c_list
    ## Assert Attributes
    assert_solved_vci_class(refVCI2,VCI2)


# = ========= JU ==== TEST ============================================= START
class RefVCI2_DAR_solved(object):
    def __init__(self):
        filename = 'test_data/data_prod/VibCI/ref_dat_VCI2_DAR.npz'
        file = np.load(filename, allow_pickle=True)
        # Attributes
        self.grids   = file['grids']
        self.wfns    = file['wfns']
        self.nmodes       = 1
        self.ngrid        = 16
        self.states       = [[0], [1], [2]]
        self.combinations = None
        self.solved       = True
        self.dx           = [7.363896585476766]
        self.a            = [0.009036093545336975]
        self.coefficients = file['coefficients']
        self.sij          = file['sij']
        self.tij          = file['tij']
        self.store_ints   = True
        self.integrals    = file['integrals'].item()
        self.store_potints = True
        int1d       = file['int1d']
        self.int1d       = int1d.item()
        self.int2d       = {}
        self.int3d       = {}
        self.int4d       = {}
        self.fortran     = False
        self.energies    = [0.00477827, 0.01426034, 0.02364815]
        self.energiesrcm = [1048.70843133, 3129.78212626, 5190.16813616]
        self.H           = file['H']
        self.vectors     = file['vectors']
        self.intensities = None
        self.maxpot      = 2
        self.v1_indices = [0,1]
        self.v1_data    = file['v1_data']
        self.v2_indices = [(0,1)]
        self.v2_data    = file['v2_data']
        self.nmax       = 2
        self.smax       = 2

class Ref_DAR_Molecule():
     # VibTools.Molecule
     def __init__(self):
         filename = 'test_data/data_prod/VibCI/Ref_DAR_VibToolsMolecule.npz'
         file = np.load(filename,allow_pickle=True)
         self.natoms =      file['natoms']
         self.atmasses =    file['atmasses']
         self.atnums =      file['atnums']
         self.coordinates = file['coordinates']



class Ref_DAR_Modes(object):
    def __init__(self):
        file = np.load('test_data/data_prod/VibCI/Ref_DAR_VibToolsModes.npz')
        self.nmodes = 1
        self.mol = None
        self.natoms = 19
        self.freqs = [2086.62515]
        modes_mw = file['modes_mw']         
        self.modes_mw = modes_mw
        modes_c = file['modes_c']
        self.modes_c = modes_c


class Ref_DAR_Grid(object):
    # VibTools.Grid
    def __init__(self):
        grids = np.load('test_data/data_prod/VibCI/Ref_DAR_VibrationsGrids_grids.npy')
        self.modes = Ref_DAR_Modes()
        self.mol = Ref_DAR_Molecule()
        self.ngrid = 16
        self.amp = 14
        self.grids = grids
        self.natoms = 19
        self.nmodes = 1

class Ref_DAR_SurfaceDipole(object):
    # VibTools.Surface.Dipole
    def __init__(self):
        # VibTools.Grids
        grids = Ref_DAR_Grid()
        self.grids = grids
        self.empty = False
        self.ngrid = 16
        self.order = 1
        self.prop = (3,)
        self.indices = [0, 1]
        data = np.load('test_data/data_prod/VibCI/Ref_DAR_VibrationsSurfaceDipole_data.npy',allow_pickle=True)
        self.data = data



@pytest.fixture
def DAR_Set(): 
    # Read SNF-Results
    path = 'test_data/DAR/'
    res = vt.SNFResults(outname=path+'snf.out',
                        restartname=path+'restart',
                         coordfile=path+'coord')
    res.read()
    subsets = np.load(path+'subsets.npy')

    ## TEST extended VibTools functions for data production
    import sys # added!
    sys.path.append("test_data/data_prod/") # added!
    import data_prod_funcs_VibTools as datprod 
    localmodes,cmat = datprod.localize_subsets(res.modes,subsets)
    #localmodes,cmat = localize_subsets(res.modes,subsets)

    ngrid = 16
    amp = 14
    grid = vib.Grid(res.mol,localmodes)
    grid.generate_grids(ngrid,amp)
    # Read ub  anharmonic 1-mode potentials
    
    v1 = vib.Potential(grid, order=1)
    v1.read_np(path+'V1_g16.npy')
    #v1.generate_harmonic(cmat=cmat)

    # Read in anharmonic 1-mode dipole moments
    dm1 = vib.Dipole(grid)
    dm1.read_np(path+'Dm1_g16.npy')
   
    # Readn in anharmonic 2-mode potentials
    v2 = vib.Potential(grid, order=2)
    v2.read_np(path+ 'V2_g16.npy')
    #v2.generate_harmonic(cmat=cmat)


    # Read in anharmonic 2-mode dipole moments    
    dm2 = vib.Dipole(grid, order=2)
    dm2.read_np(path+'Dm2_g16.npy')

    # Run VSCF calculations for these potentials
    # Here we solve only for the vibrational ground state
    
    dVSCF = vib.VSCF2D(v1,v2)
    dVSCF.solve()

    VCI2 = vib.VCI(dVSCF.get_groundstate_wfn(), v1,v2)
    VCI2.generate_states(2) # singles only
    VCI2.solve()
    return VCI2, dm1, dm2

@pytest.fixture()
def DAR_comb_generator_c():
    # for c in DAR_VCI2.combgenerator -> list
    c = [(np.array([0]), np.array([0]), 0, 0), (np.array([0]), np.array([1]), 0, 1),
              (np.array([0]), np.array([2]), 0, 2), (np.array([1]), np.array([1]), 1, 1),
              (np.array([1]), np.array([2]), 1, 2), (np.array([2]), np.array([2]), 2, 2)]
    return c

def test_calculate_transition_moments(DAR_Set):
    # Arrange
    ## Input
    VCI2, dm1, dm2 = DAR_Set
    ## Reference
    ### Attributes
    refVCI2 = RefVCI2_DAR_solved()
    filename = 'test_data/data_prod/VibCI/ref_dat_VCI2_DAR_calc_trans_moments.npz'
    file = np.load(filename,allow_pickle=True)
    refVCI2.integrals = file['integrals'].item()

    Ref_SurfDipole = Ref_DAR_SurfaceDipole()
    ### Returns
    ref_transitions = [[], 
                       [[-0.17192922, -0.33554002, -0.07779115]],
                       [[-0.00918297, -0.01871297, -0.0046235 ]]]
    # Act
    transitions = VCI2.calculate_transition_moments([dm1],[dm2])
    # Assert
    ## Returns
    assert len(transitions[0]) == 0
    np.testing.assert_almost_equal(transitions[1][0], ref_transitions[1][0])
    np.testing.assert_almost_equal(transitions[2][0], ref_transitions[2][0])
    ## Attributes
    ### VCI
    assert_solved_vci_class(refVCI2,VCI2)
    #### SurfaceDipole
    print('\n\n========================\n')
    assert_SurfaceDipoleClass(Ref_SurfDipole,VCI2.prop1[0])



def test_calculate_transition_matrix(DAR_Set):
    # Arrange
    VCI2, dm1, dm2 = DAR_Set 
    ## Reference
    refVCI2 = RefVCI2_DAR_solved()
    Ref_SurfDipole = Ref_DAR_SurfaceDipole()
    ref_path = 'test_data/data_prod/VibCI/'
    ref_transm = np.load(ref_path+'VCI_DAR_transm.npy')
    ref_freqs = np.load(ref_path+ 'VCI_DAR_freqs.npy')
    ref_inten = np.load(ref_path+ 'VCI_DAR_inten.npy') 
    # Act
    transm, inten, freqs = VCI2.calculate_transition_matrix([dm1],[dm2])
    # Assert
    ## Returns
    np.testing.assert_almost_equal(transm, ref_transm) 
    np.testing.assert_almost_equal(inten,  ref_inten) 
    np.testing.assert_almost_equal(freqs,  ref_freqs) 
    ## Attributes
    ### VCU
    assert_solved_vci_class(refVCI2,VCI2)
    ### SurfaceDipole
    assert_SurfaceDipoleClass(Ref_SurfDipole,VCI2.prop1[0])

    Ref_SurfDipole.order = 2
    Ref_SurfDipole.indices = [(0,1)]
    data = np.load(ref_path+'Ref_DAR_calc_tran_matrix_prop2_data.npy')
    Ref_SurfDipole.data = data 
    assert_SurfaceDipoleClass(Ref_SurfDipole,VCI2.prop2[0])


# = ========= JU ==== TEST ============================================= ENDE



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
    assert_SurfaceDipoleClass(ref_dm1,vci.prop1[0])

    assert vci.prop2 == None
    assert vci.transitions == None 


def test_calculate_intensities(DAR_Set):
    # Arrange
    VCI2, dm1, dm2 = DAR_Set 
    ## Reference
    refVCI2 = RefVCI2_DAR_solved()
    Ref_SurfDipole = Ref_DAR_SurfaceDipole()
    refVCI2.intensities = np.array([0., 773.07685725, 4.73254705])
    # Act
    VCI2.calculate_intensities(dm1,dm2)
    # Assert
    assert_solved_vci_class(refVCI2,VCI2,show=True)



# TODO: input: polarizability surfaces needed - Datatype?
def test_calculate_raman():
    pass 
    # Arrange
    # Act
    # Assert




# TODO: input is given by VibTools SNFResults/Ooutput ->
# Does not work...
def test_calculate_roa():
    pass 
    # Arrange
    # Act
    # Assert


#TODO: internal use functions:
#def test__v1_integral(): 
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__v2_integral():
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test_v2_integral():
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__v3_integral():
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__v4_integral():
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__ovrlp_integral(): 
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__kinetic_integral(): 
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__dgs_kinetic_integral(): 
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__dgs_ovrlp_integral(): 
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__chi(): 
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__calculate_coeff():
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__calculate_kinetic_integrals():
#    pass 
#    # Arrange
#    # Act
#    # Assert
#
#
#
#def test__calculate_ovrlp_integrals():
#    pass 
#    # Arrange
#    # Act
#    # Assert


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


@pytest.fixture
def VCISaveLoadRef():
    file = np.load('test_data/data_prod/VibCI/Ref_VCI_1_1_results.npz')
    vec = file['vec']
    enrcm = file['enrcm']
    states = file['states']
    return vec, enrcm, states


def test_save_vectors(Potentials, VCISaveLoadRef):
    # Arrange
    fname = 'VCI_1_1_results.npz'
    ## Reference
    refvec, refenrcm, refstates = VCISaveLoadRef
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    # Act 
    vci.save_vectors()
    # Assert
    fileexist = os.path.exists(fname)
    assert fileexist == True
    ## Assert saved data
    file = np.load(fname)
    vec = file['vec']
    enrcm = file['enrcm']
    states = file['states']
    np.testing.assert_almost_equal(abs(refvec),abs(vec))
    np.testing.assert_almost_equal(refenrcm,enrcm)
    np.testing.assert_almost_equal(refstates,states)
    # clean up
    os.remove(fname)



def test_read_vectors(Potentials, VCISaveLoadRef):
    # Arrange
    path = 'test_data/data_prod/VibCI/'
    fname = 'Ref_VCI_1_1_results.npz'
    ## Reference
    cm_in_au = 4.556335252760265e-06
    refvec, refenrcm, refstates = VCISaveLoadRef
    refenergies = refenrcm*cm_in_au
    ## Input
    VSCF, v1,v2 = Potentials
    gwfn =VSCF.get_groundstate_wfn()
    vci = vib.VCI(gwfn, v1,v2)
    vci.generate_states(1)
    vci.solve()
    # Act 
    vci.read_vectors(path+fname)
    # Assert
    assert vci.solved == True
    np.testing.assert_almost_equal(refvec,vci.vectors)
    np.testing.assert_almost_equal(refenrcm,vci.energiesrcm)
    np.testing.assert_almost_equal(refstates,vci.states)
    np.testing.assert_almost_equal(refenergies,vci.energies)
   

