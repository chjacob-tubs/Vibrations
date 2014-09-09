import unittest
from vibrations.Surfaces import Surface, Potential, Dipole

class TestAbstractSurface(unittest.TestCase):

    def setUp(self):
        self.surface = Surface()

    def test_empty(self):
        self.assertIs(self.surface.empty, True)

    def test_ngrid(self):
        self.assertEqual(self.surface.ngrid, 0)

    def test_order(self):
        self.assertEqual(self.surface.order, 1)
        
    def test_prop(self):
        self.assertEqual(self.surface.prop, (1,))

    def test_incides(self):
        self.assertEqual(self.surface.indices, [])

    def test_data(self):
        self.assertEqual(self.surface.data, [])

    def test_grids(self):
        self.assertIs(self.surface.grids, None)

class TestPotentialSurface(unittest.TestCase):

    def setUp(self):
        self.surface = Potential()
    
    def test_prop(self):
        self.assertEqual(self.surface.prop, (1,))

class TestDipoleSurface(unittest.TestCase):

    def setUp(self):
        self.surface = Dipole()

    def test_prop(self):
        self.assertEqual(self.surface.prop, (3,))

if __name__ == '__main__':
    unittest.main()

