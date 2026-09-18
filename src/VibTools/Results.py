# -*- coding: utf-8 -*-
#
# This file is part of the
# LocVib 1.3 suite of tools for the analysis for vibrational spectra.
# Copyright (C) 2009-2023 by Christoph R. Jacob and others.
#
#    LocVib is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    LocVib is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LocVib.  If not, see <http://www.gnu.org/licenses/>.
#
# In scientific publications using the LocVib tools, please cite:
#   Ch. R. Jacob, J. Chem. Phys 130 (2009), 084106.
#
# The most recent version of LocVib is available at
#   http://www.christophjacob.eu/software
""" Results module --> Abstract classes for interface modules."""

import numpy

from . import Constants
from .Molecule import VibToolsMolecule


class Results:
    """
    Abstract class for module interfaces : PySNF, PyXYZ.

    Attributes
    ----------
    mol : VibTools.Molecule
       Molecule class of VibTools.
    modes : VibTools.Modes
       Modes class of VibTools
    lwl : float
       laser wave length.
    """
    def __init__(self):
        """
        Results constructor.
        """
        self.mol = VibToolsMolecule()

        self.modes = None  # instance of class VibModes
        self.lwl = None

    freqs = property(lambda self: self.modes.freqs)

    def read(self):
        """Abstract method read"""
        raise Exception('Abstract method not implemented')

    def get_mw_normalmodes(self):
        """getter:  mass weighted normal modes."""
        return self.modes.modes_mw

    def get_c_normalmodes(self):
        """getter: cartesian normal modes."""
        return self.modes.modes_c

    def get_tensor_deriv_c(self, tens, ncomp=None):
        """Abstract method tensor derivative cartesian."""
        raise Exception('Abstract method not implemented')

    def get_tensor_deriv_nm(self, tens, ncomp=None, modes=None):
        deriv_c = self.get_tensor_deriv_c(tens, ncomp)
        ncomp = deriv_c.shape[2]
        if modes is None:
            modes_c = self.get_c_normalmodes()
        else:
            modes_c = modes.modes_c

        nmodes = modes_c.shape[0]
        natoms = modes_c.shape[1] // 3

        deriv_nm = numpy.zeros((nmodes, ncomp))

        for icomp in range(ncomp):
            deri_c = deriv_c[:, :, icomp].reshape((3*natoms))
            deriv_nm[:, icomp] = numpy.dot(modes_c, deri_c)
        return deriv_nm

    def get_ir_intensity(self, modes=None):
        mu = self.get_tensor_deriv_nm('dipole', modes=modes)

        irint = mu[:, 0]*mu[:, 0] + mu[:, 1]*mu[:, 1] + mu[:, 2]*mu[:, 2]

        # convert (d\mu/dR)^2 from  (au^2/amu) to (C^2 m^2 / kg)
        irint = irint * ((Constants.au_in_Debye/Constants.Bohr_in_Meter)**2
                         * (Constants.Debye_in_Cm)**2
                         * (1.0/Constants.amu_in_kg))

        # multiply by [ 1.0/(4\pi\epsilon0) * (N_A * \pi)/cvel ]
        # (Eq 13 in SNF paper)
        irint = irint * (1.0/12.0) * ((1.0/Constants.epsilon0)
                                      * Constants.Avogadro/Constants.cvel_ms)

        # divide by cvel (Eq 14 in SNF paper)
        irint = irint / (Constants.cvel_ms * 1000.0)

        # result is IR absoption in km/mol
        return irint

    def get_a2_invariant(self, gauge='len', modes=None):
        pol = self.get_tensor_deriv_nm('pol'+gauge, ncomp=6, modes=modes)
        # nmodes = pol.shape[0]  # Assigned but never used

        a2 = (1.0/3.0) * (pol[:, 0] + pol[:, 3] + pol[:, 5])
        a2 = (a2**2)*(Constants.Bohr_in_Angstrom**4)
        return a2

    def get_g2_invariant(self, modes=None):
        pol = self.get_tensor_deriv_nm('pollen', ncomp=6, modes=modes)
        # nmodes = pol.shape[0] # Assigned but never used

        g2 = (1.0/2.0) * ((pol[:, 0] - pol[:, 3])**2
                          + (pol[:, 3] - pol[:, 5])**2
                          + (pol[:, 5] - pol[:, 0])**2
                          + 6.0 * pol[:, 1]**2
                          + 6.0 * pol[:, 2]**2
                          + 6.0 * pol[:, 4]**2
                          )
        g2 = g2 * (Constants.Bohr_in_Angstrom**4)
        return g2

    def get_raman_int(self, modes=None):
        a2_inva = self.get_a2_invariant(modes=modes)
        g2_inva = self.get_g2_invariant(modes=modes)
        raman_int = 45.0*a2_inva + 7.0*g2_inva
        return raman_int

    def get_aG_invariant(self, gauge='len', modes=None):
        # for consistency with SNF alpha is always in length repr
        pol = self.get_tensor_deriv_nm('pollen', ncomp=6, modes=modes)
        gten = self.get_tensor_deriv_nm('gten'+gauge, modes=modes)

        aG = (1.0/9.0) * (pol[:, 0] + pol[:, 3] + pol[:, 5]) \
                       * (gten[:, 0] + gten[:, 4] + gten[:, 8])
        aG = aG*(Constants.Bohr_in_Angstrom**4) * (1 / Constants.cvel) * 1e6

        return aG

    def get_bG_invariant(self, gauge='len', modes=None):
        pol = self.get_tensor_deriv_nm('pol'+gauge, ncomp=6, modes=modes)
        gten = self.get_tensor_deriv_nm('gten'+gauge, modes=modes)

        bG = 0.5 * (3*pol[:, 0]*gten[:, 0] - pol[:, 0]*gten[:, 0] +
                    3*pol[:, 1]*gten[:, 1] - pol[:, 0]*gten[:, 4] +
                    3*pol[:, 2]*gten[:, 2] - pol[:, 0]*gten[:, 8] +
                    3*pol[:, 1]*gten[:, 3] - pol[:, 3]*gten[:, 0] +
                    3*pol[:, 3]*gten[:, 4] - pol[:, 3]*gten[:, 4] +
                    3*pol[:, 4]*gten[:, 5] - pol[:, 3]*gten[:, 8] +
                    3*pol[:, 2]*gten[:, 6] - pol[:, 5]*gten[:, 0] +
                    3*pol[:, 4]*gten[:, 7] - pol[:, 5]*gten[:, 4] +
                    3*pol[:, 5]*gten[:, 8] - pol[:, 5]*gten[:, 8])
        bG = bG*(Constants.Bohr_in_Angstrom**4) * (1 / Constants.cvel) * 1e6

        return bG

    def get_bA_invariant(self, modes=None):
        # always using length repr
        pol = self.get_tensor_deriv_nm('pollen', ncomp=6, modes=modes)
        aten = self.get_tensor_deriv_nm('aten', modes=modes)

        bA = 0.5 * self.lwl * ((pol[:, 3]-pol[:, 0])*aten[:, 11]
                               + (pol[:, 0]-pol[:, 5])*aten[:, 6]
                               + (pol[:, 5]-pol[:, 3])*aten[:, 15]
                               + pol[:, 1]*(aten[:, 19]-aten[:, 20]
                                            + aten[:, 8]-aten[:, 14])
                               + pol[:, 2]*(aten[:, 25]-aten[:, 21]
                                            + aten[:, 3]-aten[:, 4])
                               + pol[:, 4]*(aten[:, 10]-aten[:, 24]
                                            + aten[:, 12]-aten[:, 5])
                               )
        bA = bA*(Constants.Bohr_in_Angstrom**4) * (1 / Constants.cvel) * 1e6

        return bA

    def get_backscattering_int(self, modes=None):
        return 1e-6*96.0*(self.get_bG_invariant(gauge='vel', modes=modes)
                          + (1.0/3.0)*self.get_bA_invariant(modes=modes))
