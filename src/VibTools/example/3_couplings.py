#!/usr/bin/env python

import VibTools

from numpy import *
import os

def print_mat (a) :
    print
    for k in a :
        for l in k :
            print("%6.1f" % l,)
        print()
 
def main () :
    install_path = os.path.dirname(VibTools.__file__)
    data_path = os.path.join(install_path,"example/")

    res = VibTools.SNFResults(outname=os.path.join(data_path,'Ala10/snf.out'),
                              restartname=os.path.join(data_path,'Ala10/restart'),
                              coordfile=os.path.join(data_path,'Ala10/coord'))
    res.read()

    plots = []

    # numbers of the amide I modes
    ml = range(241,250)

    modes = res.modes.get_subset(ml)

    lv = VibTools.LocVib(modes, 'PM')
    lv.localize()
    lv.sort_by_residue()
    lv.adjust_signs()

    # vibrational coupling constants
    cmat = lv.get_couplingmat()

    print()
    print("Vibrational coupling matrix [in cm-1]")
    print_mat(cmat)

    # intensity coupling constants
    lma = VibTools.LocModeAnalysis(res, 'ROA', lv.locmodes)
    intcmat = lma.get_intensity_coupling_matrix()

    print()
    print("Intensity coupling matrix for ROA backscattering")
    print_mat(1000.0*intcmat)

main()
