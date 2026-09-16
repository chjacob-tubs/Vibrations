#!/usr/bin/env python

import VibTools

from numpy import *
import os

def main () :
    install_path = os.path.dirname(VibTools.__file__)
    data_path = os.path.join(install_path,"example/")

    res = VibTools.SNFResults(outname=os.path.join(data_path,'Ala10/snf.out'),
                            restartname=os.path.join(data_path,'Ala10/restart'),
                            coordfile=os.path.join(data_path,'Ala10/coord'))
    res.read()

    # numbers of amide I normal modes (found with composition.py)
    ml = range(241,250)

    modes = res.modes.get_subset(ml)

    backint = res.get_backscattering_int(modes=modes)

    lv = VibTools.LocVib(modes, 'PM')
    lv.localize()
    lv.sort_by_residue()
    lv.adjust_signs()

    print()
    print("Composition of localized modes: ")
    lv.locmodes.print_attype2_composition()
    lv.locmodes.print_residue_composition()

    # write localized modes to g98-style file
    lv.locmodes.write_g98out(filename="locmodes-amide1.out")

    backint_loc = res.get_backscattering_int(modes=lv.locmodes)

    print()
    print("ROA backscattering intensities of normal modes and localized modes")

    print()
    print("  normal modes        |  localized modes ")
    print("     freq       int   |    freq       int")

    n = 0
    for i, f, bi, f_loc, bi_loc in zip(ml, modes.freqs, backint*1e3,
                                           lv.locmodes.freqs, backint_loc*1e3) :
        n = n+1
        print(r"%i  %6.1f  %8.2f | %2i %6.1f %8.2f" % (i, f, bi, n, f_loc, bi_loc))

    print()
    print("Total ROA Int    : ", backint.sum() * 1e3)

main()
