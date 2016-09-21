"""
More or less useful functions and constants.

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

    s = str(s)
    l = len(s)
    l += 10

    s1 = '+'+(l-2)*'-'+'+\n'
    s2 = '|'+3*'-'+' '+s+' '+3*'-'+'|\n'

    return s1+s2+s1


import time

import cProfile

def do_cprofile(func):
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
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
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
