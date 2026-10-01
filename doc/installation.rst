************
Installation
************

The *LocVib* Python package releases with its *VibTools* modules 
are available as a GitHub repository
https://github.com/chjacob-tubs/LocVib.

*LocVib* relies on additional *Python* packages that have to be installed on the system. 
The required dependencies are defined in ``pyproject.toml``.

The main requirements are:

* **Python 3** (https://www.python.org/).

* **NumPy** (https://numpy.org/).

* **Matplotlib** (https://matplotlib.org/).

* **Open Babel 3** (https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python.html).

Additional (optional) extensions:

* **Pytest** (https://pytest.org) (recommended).

* **Sphinx** (https://www.sphinx-doc.org).

For the installation, download the repository and follow the instructions below.

.. note::
   We highly recommend using a **Conda environment** (see `Install in a Conda Environment (Recommended)`_).


.. _Download the Code:

Download the Code
=================

The *LocVib* repository can be downloaded as a ZIP file or cloned using Git.
The repository is available on GitHub at:

https://github.com/chjacob-tubs/LocVib

Download as a ZIP file
----------------------

Download the repository as a ZIP file, extract it and change to the extracted directory:

.. code-block:: console

   unzip LocVib-main.zip
   cd LocVib-main

Download using Git
------------------ 

Clone the *LocVib* repository using Git and change to the cloned directory:

.. code-block:: console

   git clone https://github.com/chjacob-tubs/LocVib.git
   cd LocVib


.. _Repository Structure:

The following installation instructions assume that you are in the root directory of the 
*LocVib* repository.

Respository Structure
=====================

After downloading the repository, the *LocVib* directory has the following structure:

.. code-block:: bash

   LocVib/
   ├── doc/
   ├── requirements/
   ├── src/
   │   └── VibTools/
   │       └── tests/
   ├── CITATION.cff   
   ├── COPYING
   ├── Dockerfile
   ├── pyproject.toml
   ├── README.md
   └── ...

The ``pyproject.toml`` file contains the package configuration, dependencies, 
and available installation variants. The source code is located in ``src/VibTools/``, 
while the documentation is located in ``doc/``.


.. _Install in a Conda Environment (Recommended):

Install in a Conda Environment (Recommended)
============================================

..
 We highly recommend using the *Conda* package manager (https://conda.io/) for the environment
 and the use of *Anaconda* (https://www.anaconda.com/) for using *Python*.
 The required Python dependencies, including *NumPy*, *Matplotlib* and
 *Open Babel*, can be installed using the Conda environment file provided
 with the repository.

 With *Pytest* we can determine the correct executability of the program.

 If we are interested in the further development of *LocVib* ourselfs,
 it makes sense to install *Sphinx* for the documentation.

We highly recommend installing *LocVib* in an isolated Conda environment.

The Conda environment can either be created manually (`Create Conda Environment`_), with the required
dependencies installed afterwards (`Installation Variants`_), or from one of the predefined environment
files provided in the ``requirements/`` directory (`Conda Environment from Requirements File`_).

The recommended way to install Conda is to follow the instructions on the 
Conda homepage (https://conda.io/).

..
 Below, we briefly describe the *Conda* commands required to install the
 necessary Python packages.

**Conda Initialization**

If Conda is installed but not yet initialized, initialize it for your shell.
For example, for ``zsh``:

.. code-block:: console

   % conda init zsh

Or, Conda can be used without initialization by running:

.. code-block:: console

   % source activate
   (base) %

.. _Create Conda Environment:

**Create Conda Environment**

If Conda is already installed and initialized, create a new environment:

.. code-block:: console

   (base) % conda create --name VibToolsCondaENV

Activate the environment:

.. code-block:: console

   (base) % conda activate VibToolsCondaENV
   (VibToolsCondaENV) % 

.. _Install LocVib with pip:

Install LocVib with pip
-----------------------

*LocVib* uses the ``pyproject.toml`` file to define the package metadata and dependencies.

Before installing *LocVib*, make sure that Python 3 and ``pip`` are available in the active environment.
If ``pip`` is not yet installed in the Conda environment, install it with:

.. code-block:: console
   
   (VibToolsCondaENV) % conda install pip

.. _Installation Variants:

**Installation Variants**

There are different predefined setup variants depending on your use case and preferred *Open Babel* package:

If you want to use the ``openbabel`` package, install with

.. code-block:: console

   (VibToolsCondaENV) LocVib % pip install ".[openbabel]"

If you want to use the ``openbabel-wheel`` package, install with

.. code-block:: console

   (VibToolsCondaENV) LocVib % pip install ".[openbabel-wheel]"

If you want to run the tests and/or build the documentation, it is recommended to do a full installation:

.. code-block:: console

   (VibToolsCondaENV) LocVib % pip install ".[full-openbabel]"

or

.. code-block:: console

   (VibToolsCondaENV) LocVib % pip install ".[full-openbabel-wheel]"

For development, *LocVib* can be installed in editable mode:

.. code-block:: console

   (VibToolsCondaENV) LocVib % pip install -e ".[full-openbabel]"

or

.. code-block:: console

   (VibToolsCondaENV) LocVib % pip install -e ".[full-openbabel-wheel]"

Verify the installation by running ``pytest`` (see `Verify Installation with Pytest`_).

.. note::
   The automatic pip installation also works without Conda environment but pip must be installed anyway.

.. _Conda Environment from Requirements File:

Conda Environment from Requirements File
----------------------------------------

The repository provides predefined Conda environments in the ``requirements/`` directory.
These environments provide the dependencies required for a basic *LocVib* installation 
and a installation including testing and documentation support.

**Basic Conda Environment**

For creating a Conda environment containing the required dependencies for *LocVib*, 
use ``requirements/environment.yml``.

.. code-block:: console
   
   (base) % cd LocVib/
   (base) LocVib % conda env create -f requirements/environment.yml
   .
   .
   .
   Downloading and Extracting Packages:

   Preparing transaction: done
   Verifying transaction: done
   Executing transaction: done
   #
   # To activate this environment, use
   #
   #     $ conda activate VibToolsCondaENV
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate

After activating the environment, *LocVib* can be installed from the root directory 
of the repository using ``pip``:

.. code-block:: console

   (base) LocVib % conda activate VibToolsCondaENV
   (VibToolsCondaENV) LocVib % pip install .
   Processing ./.
     Installing build dependencies ... done
     Getting requirements to build wheel ... done
     Preparing metadata (pyproject.toml) ... done
   .
   .
   .
   Successfully built vibtools
   Installing collected packages: vibtools
   Successfully installed vibtools-1.4

For available installation variants with *Open Babel*, see `Install LocVib with pip`_.

..   
 Now, you only need to activate the environment and add LocVib:

 .. code-block:: console

    (base) LocVib % conda activate VibToolsCondaENV
    (VibToolsCondaENV) LocVib % conda develop src/
    added /path/to/LocVib/src
    completed operation for: /path/to/LocVib/src

**Testing and Documentation Environment**

The ``requirements/environment_tests_doc.yml`` file provides a Conda environment containing 
the dependencies required for testing and documentation.

From the root directory of the *LocVib* repository, create the environment:

.. code-block:: console

   (base) % cd LocVib/
   (base) LocVib % conda env create -f requirements/environment_tests_doc.yml
   .
   .
   .
   Downloading and Extracting Packages:

   Preparing transaction: done
   Verifying transaction: done
   Executing transaction: done
   #
   # To activate this environment, use
   #
   #     $ conda activate VibToolsCondaENV
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate

After activating the environment, *LocVib* can be installed from the root directory
of the repository using ``pip``:

.. code-block:: console

   (base) LocVib % conda activate VibToolsCondaENV
   (VibToolsCondaENV) LocVib % pip install .
   Processing ./.
     Installing build dependencies ... done
     Getting requirements to build wheel ... done
     Preparing metadata (pyproject.toml) ... done
   .
   .
   .
   Building wheels for collected packages: vibtools
     Building wheel for vibtools (pyproject.toml) ... done
     Created wheel for vibtools: filename=vibtools-1.4-py3-none-any.whl size=8387926 sha256=cb54ca71ee4b945387fec40e19f3057e227ffadafbe4c835b77c3568d16eea60
     Stored in directory: /tmp/pip-ephem-wheel-cache-l__b64y1/wheels/cb/e8/46/a1a2c19b51945c091486fc56b252aa16e49e57c6b84350239f
   Successfully built vibtools
   Installing collected packages: ...
   Successfully installed vibtools-1.4

For available installation variants with *Open Babel*, see `Install LocVib with pip`_.

Additional documentation packages can be installed if required:

.. code-block:: console

   (VibToolsCondaENV) LocVib % pip install docstr-coverage sphinx-mdinclude

Verify the installation by running ``pytest`` (see `Verify Installation with Pytest`_).
The required testing dependencies, including ``pytest``, are already installed through the
provided ``requirements/environment_tests_doc.yml``.


.. _Install in a Python Virtual Enviroment:

Install in a Python Virtual Environment
=======================================

Alternatively, a Python virtual environment can be created using the ``venv`` module:

.. code-block:: console

   python -m venv venv-locvib

Activate the virtual environment:

.. code-block:: console

   source venv-locvib/bin/activate

After activation, *LocVib* can be installed using ``pip``. 

For a basic installation:

.. code-block:: console 

   (venv-locvib) LocVib % pip install ".[openbabel]"

For a full installation, including tests and documentation:

.. code-block:: console

   (venv-locvib) LocVib % pip install ".[full-openbabel]"

For other installation variants, see `Install LocVib with Pip`_.

Verify the installation by running ``pytest`` (see `Verify Installation with Pytest`_).


.. _Manual Installation (PYTHONPATH):

Manual Installation (PYTHONPATH)
================================

The subdirectory ``LocVib/src/VibTools`` has to be included in the
PYTHONPATH environment variable.

The *.zprofile* file can be modified by adding:

.. code-block:: console

   export PYTHONPATH="${PYTHONPATH}:/path/to/LocVib/src/"

Or, the following lines can be added to a script before importing *LocVib*:

.. code-block:: console

   import sys
   sys.path.append('/path/to/LocVib/src/')


.. _Verify Installation with Pytest:

Verify Installation with Pytest
===============================

Verify the installation by running ``pytest`` (works in testing and full installations only):

.. code-block:: console

   src/VibTools/tests % pytest -v

.. note::
   Another option to check the executability of the program is to calculate the code examples. 
   
   The examples are located in ``LocVib/src/VibTools/example/``.
   
   See also:

   .. toctree::

      examples


.. _Documentation:

Documentation
=============

You can find the documentation, as mentioned above,
on the homepage (https://locvibvibtools.readthedocs.io/en/latest/)
or you can generate the HTML documentation yourself with Sphinx.

Build documentation via Sphinx and extension packages (works in docs installations only):

.. code-block:: console

   doc/ % sphinx-build . build

Opening the ``index.html`` file takes you to the home directory of the code documentation:

.. code-block:: console

   doc/build/ % open index.html

.. note:: The documentation is automatically built and deployed as defined in ``.gitlab-ci.yml``.

