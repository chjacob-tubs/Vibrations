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

# The most recent version of LocVib is available at
#   http://www.christophjacob.eu/software

"""
 Constants used in program.
"""

pi = 3.141592653589793
"""float: pi,the ratio of a circle's circumference to its diameter."""

cvel = 137.0359895
"""float: speed of light in atomic units."""

cvel_ms = 2.99792458e08
"""float: speed of light in meter per second."""

Bohr_in_Angstrom = 0.5291772108
"""float: conversion from Bohr to Angstrom"""

Bohr_in_Meter = Bohr_in_Angstrom * 1.0e-10
"""float: conversion from Bohr to meter. """

Avogadro = 6.02214199e23
"""float: The Avogadro constant in 1/mol."""

amu_in_kg = 1.0e-3/Avogadro
"""float: conversion atomic units in kg."""

Hartree_in_Joule = 4.35974381e-18
"""float: conversion from Hartree in Joule"""

eV_in_Joule = 1.6021765654e-19
"""float: conversion from eV (electronvolt) in Joule."""

au_in_Debye = 2.54177
"""float: conversion vrom automic units in Debye."""

Debye_in_Cm = 3.33564e-30
"""float: conversion from Debye in Cm."""

epsilon0 = 8.854187817e-12  # in SI units
"""float: Vacuum permittivity in SI units."""

h_SI = 6.62606957e-34  # in SI units
"""float: Planck constant in SI units."""

me_in_amu = 5.4857990943e-4   # mass of electron in amu
"""float: conversion from mass of electron in atomic units."""

atu_in_s = 2.41888432650516e-17  # atomic time unit in seconds
"""float: conversion from atomic units in seconds"""


cm_in_au = atu_in_s * (2.0*pi*1e2*cvel_ms)   # cm-1 -> au
"""float: conversion from 1/cm in atomic units."""

intfactor = 2.5066413842056297
""" factor to calculate integral absorption coefficient having freq in [cm-1]\n
and dipole moment in [Debye]."""
