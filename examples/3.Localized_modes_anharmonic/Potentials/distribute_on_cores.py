#!/usr/bin/env python
# ./distribute_on_cores.py <how_many_cores> <jobrunner>
#

from glob import glob
import sys
import shutil
import os

nodes = int(sys.argv[1])
runner = sys.argv[2]

endir = 'energies'

print 'Cores: ', nodes
print 'Runner:', runner
print 'Energies stored to: ', endir
os.mkdir(endir)
os.chdir(endir)
endirpath = os.getcwd()
os.chdir('..')

for i in range(nodes):
    dirname = str(i+1)
    os.mkdir(dirname)
    shutil.copy(runner,dirname)
    shutil.copy('restart', dirname)
    shutil.copy('snf.out', dirname)
    shutil.copy('coord', dirname)
    os.chdir(dirname)
    f = open(runner)
    lines = f.readlines()
    for i,l in enumerate(lines):
        if 'actcore' in l:
            lines[i] = 'actcore = ' + dirname + '\n'
            break
    for i,l in enumerate(lines):
        if 'totcores' in l:
            lines[i] = 'totcores = ' + str(nodes) + '\n'
            break
    for i,l in enumerate(lines):
        if 'enpath = ' in l:
            lines[i] = 'enpath = \'' + endirpath + '\'\n'
            break
    f.close()
    f = open(runner,'w')
    for l in lines:
        f.write(l)

    f.close()
    os.chdir('..')
    
#for i,n in enumerate(glob(where+'/*.xyz')):
#    dest = i%nodes + 1
    #print 'File %s goes to %i' %(n,dest)
    #shutil.copy(n,str(dest))
#    shutil.move(n,str(dest))

