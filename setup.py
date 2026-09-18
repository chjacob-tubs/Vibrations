from setuptools import setup
from setuptools.command.build_py import build_py
import subprocess
import os
import sys
import glob
import shutil

class BuildWithFints(build_py):
    def run(self):
        # Check if 'fints' is in the parent process command line
        try:
            with open(f'/proc/{os.getppid()}/cmdline', 'rb') as f:
                cmdline = f.read().replace(b'\x00', b' ').decode()
        except Exception:
            cmdline = ""

        if 'fints' in cmdline:
            print("Compiling Fortran libraries...")
            fortran_dir = os.path.join(os.path.dirname(__file__), 'src', 'Vibrations')

            # Get all .f files
            fortran_files = glob.glob(os.path.join(fortran_dir, 'v*int.f'))
            if not fortran_files:
                raise RuntimeError(f"No Fortran files found matching v*int.f in {fortran_dir}")

            # Compile with f2py
            cmd = [sys.executable, '-m', 'numpy.f2py', '-c'] + fortran_files + ['-m', 'fints']
            subprocess.check_call(cmd, cwd=fortran_dir)

            # Find the compiled .so file (e.g., fints.cpython-313-x86_64-linux-gnu.so)
            so_files = glob.glob(os.path.join(fortran_dir, 'fints*.so'))
            if not so_files:
                raise RuntimeError("Compiled .so file not found after f2py build.")

            compiled_so = so_files[0]
            print(f"Found compiled file: {compiled_so}")

            # Copy it to fints.so
            target_so = os.path.join(fortran_dir, 'fints.so')
            if os.path.exists(target_so):
                os.remove(target_so)  # Remove old copy
            print(f"Copying {compiled_so} → {target_so}")
            shutil.copy2(compiled_so, target_so)

#            # Move to package directory
#            package_dir = os.path.join(os.path.dirname(__file__), 'src', 'Vibrations')
#            target_in_package = os.path.join(package_dir, 'fints.so')
#            if os.path.exists(target_in_package):
#                os.remove(target_in_package)
#            print(target_so)
#            print(target_in_package)
#            shutil.copy2(target_so, target_in_package)
#
#            print(f"Successfully copied: {target_in_package}")
        else:
            print("fints not requested. Skipping Fortran compilation.")

        super().run()


setup(cmdclass={'build_py': BuildWithFints})
