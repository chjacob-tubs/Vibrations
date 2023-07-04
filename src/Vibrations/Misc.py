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
Module containing more or less useful functions and constants.

@var pi: the ratio of a circle's circumference to its diameter
@var cvel: speed of light in  atomic units
@var cvel_ms: speed of light / m*s**-1
@var Bohr_in_Angstrom: length unit conversion factor Bohr -> Angstrom
@var Bohr_in_Meter: length unit conversion factor Bohr -> Meter
@var Avogadro: the Avogadro's constant / mol**-1
@var amu_in_kg: atomic mass unit [u] in kilograms
@var Hartree_in_Joule: energy unit conversion factor
@var eV_in_Joule: energy unit conversion factor
@var au_in_Debye: dipole moment unit conversion factor atomic units -> Debye
@var Debye_in_Cm: dipole moment unit conversion factor Debye -> Coulomb * meter
@var epsilon0: vacuum permittivity in SI units
@var h_SI: Planck's constant in SI units
@var me_in_amu: mass of electron in atomic mass unit me -> u
@var atu_in_s: time unit conversion factor atomic unit of time -> second
@var cm_in_au: energy conversion unit cm**-1 -> atomic unit of energy
@var intfactor: Infrared integral absorption prefactor
"""


def fancy_box(s):  # doing a fancy box around a string
    """
    Just printing a box around a string.

    Parameters
    ----------
    s : all data types
        all data types are allowed.
    """

    s = str(s)
    s = s.strip()
    l = len(s)
    l += 10

    s1 = '+'+(l-2)*'-'+'+\n'
    s2 = '|'+3*'-'+' '+s+' '+3*'-'+'|\n'

    return s1+s2+s1


import time

import cProfile

def do_cprofile(func):
    """
    Profiler decorator

    cProfile provides deterministic profiling of Python programs. 
    A profile is a set of statistics that describes how often and 
    for how long various parts of the program executed.
    
    Example:
    
    @vibrations.do_cprofile
    def example_function():  
        return 1

    Parameters
    ----------
    func : arbitrary function
       You can use an arbitrary function. 
    """
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

def timefunc(f):
    """
    Timing decorator

    Shows the elapsed time to execute a function.

    @vibrations.timefunc
    def example_function():  
        return 1

    Parameters
    ----------
    func : arbitrary function
       You can use an arbitrary function. 
    """
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, 'took', end - start, 'time')
        return result
    return f_timer

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

pi = 3.141592653589793
cvel = 137.0359895
cvel_ms = 2.99792458e08

# conversion from bohr to Angstrom
Bohr_in_Angstrom = 0.5291772108
Bohr_in_Meter = Bohr_in_Angstrom * 1.0e-10

Avogadro = 6.02214199e23

amu_in_kg = 1.0e-3/Avogadro

Hartree_in_Joule = 4.35974381e-18
eV_in_Joule = 1.6021765654e-19


au_in_Debye = 2.54177
Debye_in_Cm = 3.33564e-30

epsilon0 = 8.854187817e-12  # in SI units
h_SI = 6.62606957e-34  # in SI units

me_in_amu = 5.4857990943e-4   # mass of electron in amu
atu_in_s = 2.41888432650516e-17  # atomic time unit in seconds

cm_in_au = atu_in_s * (2.0*pi*1e2*cvel_ms)   # cm-1 -> au

intfactor = 2.5066413842056297  # factor to calculate integral absorption coefficient having  [cm-1]  and  [Debye]
