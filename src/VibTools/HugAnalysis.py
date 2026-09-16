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
"""
HugAnalysis module.
Features for analyzing calculated vibrational spectra.

Further information:

C. R. Jacob, S. Luber, M. Reiher, Chem. Eur. J. 2009, 15, 13491-13508.

C. R. Jacob, S. Luber, M. Reiher, J. Phys. Chem. B, 2009, 113 (18), 6558-6573.
"""

import numpy

from . import Constants
# from . import Modes


class HugAnalysis:
    """
    features for analyzing calculated vibrational spectra.

    Attributes = Parameters

    Parameters
    ----------
    res : VibTools.PySNFResults
       Results of PYSNF.
    tens : str
       tensor name(ROA, Raman, IR, a2, g2, aG, bG, bA, id)
    scale : float
       scaling factor.
    """

    def __init__(self, res, tensor, scale=1.0):
        """
        HugAnalysis constructor.
        For more details see the class description/docstring
        """

        self.natoms = res.modes.natoms
        if tensor == 'ROA':
            self.tensor_decomposed_c = self.get_backint_decomposed_c(res)
        elif tensor == 'Raman':
            self.tensor_decomposed_c = self.get_ramanint_decomposed_c(res)
        elif tensor == 'IR':
            self.tensor_decomposed_c = self.get_irint_decomposed_c(res)
        elif tensor in ['a2', 'g2', 'aG', 'bG', 'bA']:
            self.tensor_decomposed_c = eval('self.get_' +
                                            tensor + '_decomposed_c(res)')
        elif tensor == 'id':
            self.tensor_decomposed_c = numpy.ones((self.natoms*3,
                                                   self.natoms*3))

        self.tensor_decomposed_c = scale * self.tensor_decomposed_c

    def get_irint_decomposed_c(self, res):
        """
        gets IR intensities (decomposed/cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.

        Returns
        -------
        mu : numpy.ndarray
          IR intensities.
        """
        dip = res.get_tensor_deriv_c('dipole')
        dip = dip.reshape((self.natoms*3, 3))

        mu = (numpy.outer(dip[:, 0], dip[:, 0]) +
              numpy.outer(dip[:, 1], dip[:, 1]) +
              numpy.outer(dip[:, 2], dip[:, 2]))
        # TODO:  the input parameter "res" already exists
        #        as an class attribute.
        # FIXME: convert to absorption (in km/mol);
        #        scale factor stolen from SNF
        mu = mu * 863.865928384
        return mu

    def get_a2_decomposed_c(self, res, gauge='len'):
        """
        gets a2 polaribilities  (decomposed,cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.
        gauge : str
           To ensure gauge invariance, the velocity representation \
           of the electric-dipole operator.

        Results
        -------
        a2 : numpy.ndarray
           a2 polarizability tensor.
        """
        pol = res.get_tensor_deriv_c('pol'+gauge, 6)

        temp = (1.0/3.0)*(pol[:, :, 0] + pol[:, :, 3] + pol[:, :, 5])
        temp = temp.reshape((self.natoms*3,))

        a2 = numpy.outer(temp[:], temp[:])
        a2 = a2*(Constants.Bohr_in_Angstrom**4)
        return a2

    def get_g2_decomposed_c(self, res):
        """
        get g2 polarizibility tensor (decomposed, cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.

        Results
        -------
        g2 : numpy.ndarray
        g2 polarizibility tensor.
        """

        pol = res.get_tensor_deriv_c('pollen', 6)

        pol = pol.reshape((self.natoms*3, 6))

        g2 = 3.0*(numpy.outer(pol[:, 0], pol[:, 0])
                  + numpy.outer(pol[:, 1], pol[:, 1])
                  + numpy.outer(pol[:, 2], pol[:, 2])
                  + numpy.outer(pol[:, 1], pol[:, 1])
                  + numpy.outer(pol[:, 3], pol[:, 3])
                  + numpy.outer(pol[:, 4], pol[:, 4])
                  + numpy.outer(pol[:, 2], pol[:, 2])
                  + numpy.outer(pol[:, 4], pol[:, 4])
                  + numpy.outer(pol[:, 5], pol[:, 5]))
        g2 = g2 - numpy.outer(pol[:, 0]+pol[:, 3]+pol[:, 5],
                              pol[:, 0]+pol[:, 3]+pol[:, 5])

        g2 = 0.5*g2*(Constants.Bohr_in_Angstrom**4)

        return g2

    def get_ramanint_decomposed_c(self, res):
        """
        gets raman intensities (decomposed, cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.

        Results
        -------
        ramanint_decomp_c : numpy.ndarray
        raman intensities.
        """
        a2_decomp_c = self.get_a2_decomposed_c(res)
        g2_decomp_c = self.get_g2_decomposed_c(res)
        ramanint_decomp_c = 45.0*a2_decomp_c + 7.0*g2_decomp_c
        return ramanint_decomp_c

    def get_aG_decomposed_c(self, res, gauge='len'):
        """
        gets aG tensor (decomposed, cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.
        gauge : str
            ensure gauge invariance (length).

        Returns
        -------
        aG : numpy.ndarray
        aG polarizibility tensor.
        """

        # for consistency with SNF alpha is always in length repr
        pol = res.get_tensor_deriv_c('pollen', 6)
        gten = res.get_tensor_deriv_c('gten'+gauge)

        temp1 = (1.0/3.0)*(pol[:, :, 0] + pol[:, :, 3] + pol[:, :, 5])
        temp1 = temp1.reshape((self.natoms*3,))
        temp2 = (1.0/3.0)*(gten[:, :, 0] + gten[:, :, 4] + gten[:, :, 8])
        temp2 = temp2.reshape((self.natoms*3,))

        aG = numpy.outer(temp1[:], temp2[:])
        aG = aG*(Constants.Bohr_in_Angstrom**4) * (1/Constants.cvel) * 1e6

        return aG

    def get_bG_decomposed_c(self, res, gauge='len'):
        """
        gets bG tensor (decomposed, cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.
        gauge : str
            ensure gauge invariance (length).

        Returns
        -------
        bG : numpy.ndarray
        bG polarizibility tensor.
        """

        pol = res.get_tensor_deriv_c('pol'+gauge, 6)
        gten = res.get_tensor_deriv_c('gten'+gauge)

        pol = pol.reshape((self.natoms*3, 6))
        gten = gten.reshape((self.natoms*3, 9))

        bG = 3.0*(numpy.outer(pol[:, 0], gten[:, 0])
                  + numpy.outer(pol[:, 1], gten[:, 1])
                  + numpy.outer(pol[:, 2], gten[:, 2])
                  + numpy.outer(pol[:, 1], gten[:, 3])
                  + numpy.outer(pol[:, 3], gten[:, 4])
                  + numpy.outer(pol[:, 4], gten[:, 5])
                  + numpy.outer(pol[:, 2], gten[:, 6])
                  + numpy.outer(pol[:, 4], gten[:, 7])
                  + numpy.outer(pol[:, 5], gten[:, 8]))
        bG = bG - numpy.outer(pol[:, 0] + pol[:, 3]+pol[:, 5],
                              gten[:, 0] + gten[:, 4] + gten[:, 8])

        bG = 0.5*bG*(Constants.Bohr_in_Angstrom**4) * (1/Constants.cvel) * 1e6

        return bG

    def get_bA_decomposed_c(self, res):
        """
        gets aA tensor (decomposed, cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.

        Returns
        -------
        bA : numpy.ndarray
        bA polarizibility tensor.
        """
        pol = res.get_tensor_deriv_c('pollen', 6)
        aten = res.get_tensor_deriv_c('aten')

        pol = pol.reshape((self.natoms*3, 6))
        aten = aten.reshape((self.natoms*3, 27))

        bA = (numpy.outer(pol[:, 3]-pol[:, 0], aten[:, 11])
              + numpy.outer(pol[:, 0]-pol[:, 5], aten[:, 6])
              + numpy.outer(pol[:, 5]-pol[:, 3], aten[:, 15])
              + numpy.outer(pol[:, 1], aten[:, 19] -
                            aten[:, 20]+aten[:, 8] - aten[:, 14])
              + numpy.outer(pol[:, 2], aten[:, 25] - aten[:, 21] +
                            aten[:, 3]-aten[:, 4])
              + numpy.outer(pol[:, 4], aten[:, 10] -
                            aten[:, 24]+aten[:, 12]-aten[:, 5])
              )
        bA = (0.5 * res.lwl * bA * (Constants.Bohr_in_Angstrom**4) *
              (1/Constants.cvel) * 1e6
              )
        return bA

    def get_backint_decomposed_c(self, res):
        """
        gets backbone intensities (decomposed, cartesian components).

        Parameters
        ----------
        res : VibTools.PySNFResults
           Results of PYSNF.

        Returns
        -------
        backint_decomp_c : numpy.ndarray
           backbone intensities.
        """
        pre_factor = 1e-6*96.0
        bG_decomp_c = self.get_bG_decomposed_c(res, gauge='vel')
        bA_decomp_c = self.get_bA_decomposed_c(res)
        backint_decomp_c = pre_factor*(bG_decomp_c+(1.0/3.0)*bA_decomp_c)
        return backint_decomp_c

    def project_on_modes(self, modes, nummode=None):
        """
        projects given tensor on modes.

        Parameters
        ----------
        modes : VibTools.Modes
           VibTools modes class.
        nummode : None or list
           numbering of modes.

        Returns
        -------
        inv_decomposed_nm : numpy.ndarray
        tensors projected on normal modes (inverted).
        """
        inv_decomposed_nm = numpy.zeros((self.natoms, self.natoms))

        if nummode is None:
            modelist = list(range(modes.nmodes))
        else:
            modelist = [nummode]

        for imode in modelist:
            # projection on normal modes
            temp = self.tensor_decomposed_c * numpy.outer(modes.modes_c[imode],
                                                          modes.modes_c[imode])

            # now sum x, y, z-components (in both direction)
            temp = temp.reshape((self.natoms, 3, self.natoms, 3))
            temp = temp[:, :, :, 0] + temp[:, :, :, 1] + temp[:, :, :, 2]
            temp = temp[:, 0, :] + temp[:, 1, :] + temp[:, 2, :]

            inv_decomposed_nm += temp

        return inv_decomposed_nm

# TODO siehe paper: Analysis of Secondary Structure Effects
# on the IR and Raman Spectra of Polypeptides in Terms of Localized Vibrations
    def sum_groups(self, inv_decomposed_nm, groups):
        """
        sums groups.

        Parameters
        ----------
        inv_decomposed_nm : numpy.ndarray
           invariant decomposed normal modes.
        groups : list of lists.
            example: [[], [], [], [], [], [], [], [], [], [], [0, 1, 2]]

        Returns
        -------
        inv_groups : list
        invariant groups.
        """
        ngroups = len(groups)
        inv_groups = numpy.zeros((ngroups, ngroups))

        for ig in range(ngroups):
            for jg in range(ngroups):
                for i in groups[ig]:
                    for j in groups[jg]:
                        inv_groups[ig, jg] += inv_decomposed_nm[i, j]

        inv_groups = inv_groups + inv_groups.transpose()

        for i in range(ngroups):
            inv_groups[i, i] = inv_groups[i, i] / 2
            inv_groups[i, :i] = 0.0

        return inv_groups

    def get_group_coupling_matrix(self, groups, modes, num_mode=None):
        """
        gets group coupling matrix.

        Parameters
        ----------
        groups : list of list
            example: [[], [], [], [], [], [], [], [], [], [], [0, 1, 2]]
        modes : VibTools.Modes
           VibTools modes class.
        num_mode : None or list
           numbering of modes.

        Returns
        -------
        inv_groups : list
        invariant groups.
        """
        inv_decomposed_nm = self.project_on_modes(modes, num_mode)
        inv_groups = self.sum_groups(inv_decomposed_nm, groups)
        return inv_groups

    def print_gcm(self, inv_groups, groupnames):
        """
        prints gcm.

        Parameters
        ----------
        inv_groups : list
           invariant groups.
        groupnames : str
            group names:'ROA','Raman','IR','a2','g2','aG','bG','bA','id'
        """
        for n in groupnames:
            print(("%6s " % n), end=' ')
        print()
        for i in range(inv_groups.shape[0]):
            print(" "*8*i, end=' ')
            for j in range(i, inv_groups.shape[0]):
                print("%6.1f " % inv_groups[i, j], end=' ')
            print()

    def print_group_coupling_matrix(self, groups, groupnames, modes,
                                    num_mode=None, scale=1.0):
        """
        prints group coupling matrix.

        Parameters
        ----------
        groups : list of list
            example: [[], [], [], [], [], [], [], [], [], [], [0, 1, 2]]
        groupnames : str
            group names:'ROA','Raman','IR','a2','g2','aG','bG','bA','id'
        modes : VibTools.Modes
           VibTools modes class.
        num_mode : None or list
           numbering of modes.
        """
        inv_groups = self.get_group_coupling_matrix(groups, modes, num_mode)
        print()
        print("Total intensity: ", inv_groups.sum())
        print()
        self.print_gcm(inv_groups*scale, groupnames)


class LocModeAnalysis(HugAnalysis):
    """
    features for analyzing calculated vibrational spectra
    in terms of localized modes.

    Attributes inherites from HugAnalysis class.
    """
    def __init__(self, res, tensor, locmodes, scale=1.0):
        """
        LocModeAnalysis constructor.
        """
        HugAnalysis.__init__(self, res, tensor, scale)
        self.locmodes = locmodes
        self.nmodes = locmodes.nmodes
        self.tensor_decomposed_lm = self.get_tensor_decomposed_lm()

    def get_tensor_decomposed_lm(self):
        """
        gets decomposed local modes tensor.

        Returns
        -------
        tens_decomposed_lm : numpy.ndarray
        """
        tens_decomposed_lm = numpy.zeros((self.nmodes, self.nmodes))

        for imode in range(self.nmodes):
            for jmode in range(self.nmodes):
                temp = self.tensor_decomposed_c * \
                       numpy.outer(self.locmodes.modes_c[imode],
                                   self.locmodes.modes_c[jmode])

                tens_decomposed_lm[imode, jmode] = temp.sum()

        tens_decomposed_lm = (tens_decomposed_lm
                              + tens_decomposed_lm.transpose())

        for i in range(self.nmodes):
            tens_decomposed_lm[i, i] = tens_decomposed_lm[i, i] / 2
            tens_decomposed_lm[i, :i] = 0.0

        return tens_decomposed_lm

    def get_intensity_coupling_matrix(self, mode=None):
        """
        gets intensity coupling matrix.

        Returns
        -------
        tens_decomposed_lm : numpy.ndarray
        """
        if mode is None:
            return self.tensor_decomposed_lm
        else:
            return self.tensor_decomposed_lm * numpy.outer(mode, mode)
