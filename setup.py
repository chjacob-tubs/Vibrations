import setuptools

setuptools.setup(
      include_package_data = True,
      name         = 'Vibrations',
      version      = '1.0',
      description  = 'Vibrations is a Python code for vibrational calculations with user-defined vibrational modes',
      author       = 'Pawel T. Panek, Christoph Jacob, Julia Brueggemann, Mario Wolter, Michael Welzel and others',
      url          = 'https://www.tu-braunschweig.de/pci/agjacob/software',
      license      = 'GPLv3',
      package_dir  = {'': 'src/'},
      python_requires = '~=3.11.4',
      install_requires = ['numpy~=1.23.4',
                          'matplotlib~=3.6.1',
                          'scipy~=1.9.3',
                          'openbabel-wheel~=3.1.1.16'],
      classifiers  = ["Programming Language :: Python :: 3",
                      "Operating System :: OS Independent"],
     )

