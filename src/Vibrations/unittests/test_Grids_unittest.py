#!/usr/bin/env python

import unittest
from Vibrations.Grids import Grid

class TestEmptyGrid(unittest.TestCase):

    def setUp(self):
        self.emptygrid = Grid()

    def test_natoms(self):
        self.assertEqual(self.emptygrid.natoms,0)

    def test_nmodes(self):
        self.assertEqual(self.emptygrid.nmodes,0)

    def test_ngrid(self):
        self.assertEqual(self.emptygrid.ngrid,0)

    def test_amp(self):
        self.assertEqual(self.emptygrid.amp,0)

    def test_mol(self):
        self.assertIs(self.emptygrid.mol,None)

    def test_modes(self):
        self.assertIs(self.emptygrid.modes,None)

    def test_generate_grids(self):
        self.assertRaises(Exception,self.emptygrid.generate_grids,-1,-2)
        self.assertRaises(Exception,self.emptygrid.generate_grids)
        self.assertRaises(Exception,self.emptygrid.generate_grids,0,0)
        self.assertRaises(Exception,self.emptygrid.generate_grids,0.5,0.1)
        self.assertRaises(Exception,self.emptygrid.generate_grids,1,0.1)
        self.assertRaises(Exception,self.emptygrid.generate_grids,0.1,1)

    def test_get_grid_structure(self):
        self.assertRaises(Exception, self.emptygrid.get_grid_structure)
        self.assertRaises(Exception, self.emptygrid.get_grid_structure,[0],[0])
        self.assertRaises(Exception, self.emptygrid.get_grid_structure,[0],[0,2])
        self.assertRaises(Exception, self.emptygrid.get_grid_structure,[0,1],[0])

    def test_get_pyadf_molecule(self):
        self.assertRaises(Exception, self.emptygrid.get_pyadf_molecule)
        self.assertRaises(Exception, self.emptygrid.get_pyadf_molecule,[0],[0])
        self.assertRaises(Exception, self.emptygrid.get_pyadf_molecule,[0],[0,2])
        self.assertRaises(Exception, self.emptygrid.get_pyadf_molecule,[0,1],[0])

    def test_read_np(self):
        self.assertRaises(Exception, self.emptygrid.read_np)
        self.assertRaises(Exception, self.emptygrid.read_np,'notexistingfile')

    def test_get_number_of_grid_points(self):
        self.assertEqual(self.emptygrid.get_number_of_grid_points(),0)

    def test_get_number_of_modes(self):
        self.assertEqual(self.emptygrid.get_number_of_modes(), 0)

    def test_save_grids(self):
        import numpy as np
        import os

        self.emptygrid.save_grids()
        fname = 'grids_' + str(self.emptygrid.ngrid) + '_' + str(self.emptygrid.amp) + '.npy'
        tmp = np.load(fname)
        os.remove(fname)
        self.assertTrue(np.array_equal(self.emptygrid.grids,tmp))

if __name__ == '__main__':
    unittest.main()
#suite = unittest.TestLoader().loadTestsFromTestCase(TestEmptyGrid)
#unittest.TextTestRunner(verbosity=2).run(suite)

