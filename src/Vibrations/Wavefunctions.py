# This file is a part of 
# Vibrations - a Python Code for Anharmonic Theoretical Vibrational Spectroscopy
# Copyright (C) 2014-2023 by Pawel T. Panek, Adrian A. Hoeske, Julia Brüggemann,
# Michael Welzel, and Christoph R. Jacob.
#
#    Vibrations is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Vibrations is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Vibrations.  If not, see <http://www.gnu.org/licenses/>.
#
# In scientific publications using Vibrations please cite:
#   P. T. Panek, Ch. R. Jacob, ChemPhysChem 15 (2014) 3365.
#   P. T. Panek, Ch. R. Jacob, J. Chem. Phys. 144 (2016) 164111.
# 
# The most recent version of Vibrations is available at
#   http://www.christophjacob.eu/software
"""
Module related to the vibrational wave functions class
"""
import numpy as np


class Wavefunction(object):
    """
    The class containing and manipulating the vibrational wave functions.

    The class can be initialized with the grids. If none are given, an empty object is created, which can be used
    to read in an existing wave function.

    Parameters
    ----------
    nmodes : int
       number of modes.
    grids : Vibrations/Grid 
       The grids (Vibrations.Grid).
    ngrid : int
       number of grid points.
    nstates : int 
       number of states.
    wfns : ndarray
       wavefunctions.
    """
    def __init__(self, grids=None):
        """
        Wavefunction constructor.
        Further details in class description.
        """
        if grids is None:
            self.nmodes = 0
            self.ngrid = 0
            self.nstates = 0
            self.wfns = np.zeros((self.nmodes, self.nstates, self.ngrid))

        else:
            self.nmodes = grids.nmodes
            self.ngrid = grids.ngrid
            self.nstates = self.ngrid
            self.wfns = np.zeros((self.nmodes, self.nstates, self.ngrid))

    def save_wavefunctions(self, fname='wavefunctions'):
        """
        Saves the wave function to a NumPy formatted binary file `*.npy`.

        Parameters
        ----------
        fname : Str 
           File Name
        """
        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.wfns)

    def read_wavefunctions(self, fname='wavefunctions.npy'):
        """
        Reads in existing vibrational wave functions from a NumPy formatted binary file `*.npy`.

        Parameters
        ----------
        fname : Str 
           File Name
        """
        tmpwfns = np.load(fname)
        self.nmodes = tmpwfns.shape[0]
        self.ngrid = tmpwfns.shape[2]
        self.nstates = tmpwfns.shape[1]
        self.wfns = tmpwfns.copy()
