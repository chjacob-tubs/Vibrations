.. _Installation:

************
Installation
************

The Vibrations Python package releases are available as a GitHub repository
https://github.com/chjacob-tubs/Vibrations.

Vibrations relies on additional Python packages that have to be installed on the system.
The required dependencies are defined in the ``pyproject.toml`` file.

The main requirements are:

* **Python 3** (https://www.python.org/).

* **NumPy** (https://numpy.org/).

* **Matplotlib** (https://matplotlib.org/).

* **Open Babel 3** (https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python.html).

Non-standard packages:

* **LocVib** (https://github.com/chjacob-tubs/LocVib).

* **PyADF** (https://github.com/chjacob-tubs/pyadf-releases).

Additional (optional) extensions:

* **Pytest** (https://pytest.org) (recommended).

* **Sphinx** (https://www.sphinx-doc.org).

For the installation, download the repository and follow the instructions below.

.. note::
   We highly recommend using a **Conda environment** (see `Install in a Conda Environment (Recommended)`_).

.. _Download the Code:

Download the Code
=================

The Vibrations repository can be downloaded as a ZIP file or cloned using Git.
The repository is available on GitHub at:

https://github.com/chjacob-tubs/Vibrations

Download as a ZIP File
----------------------

Download the repository as a ZIP file, extract it and change to the extracted directory:

.. code-block:: console

   unzip Vibrations-main.zip
   cd Vibrations-main

Download using Git
------------------

Clone the Vibrations repository using Git and change to the cloned directory:

.. code-block:: console

   git clone https://github.com/chjacob-tubs/Vibrations.git
   cd Vibrations

.. _Repository Structure:

The following installation instructions assume that you are in the root directory of the
Vibrations repository.

Respository Structure
=====================

After downloading the repository, the Vibrations directory has the following structure:

.. code-block:: bash

   Vibrations/
   ├── doc/
   ├── examples/
   ├── requirements/
   ├── src/
   │   └── Vibrations/
   │       └── tests/
   │       └── unittests/
   ├── CITATION.cff
   ├── Dockerfile
   ├── Dockerfile.Conda
   ├── Dockerfile.sphinx_doc
   ├── Dockerfile.unittests
   ├── LICENSE
   ├── pyproject.toml
   ├── README.md
   ├── readthedocs.yml
   ├── setup.py
   └── ...

The ``pyproject.toml`` file contains the package configuration, dependencies,
and available installation variants. The source code is located in ``src/Vibrations/``,
while the documentation is located in ``doc/``. ``tests/`` and ``unittests/`` contain 
the tests for the Vibrations package.

.. _Install in a Conda Environment (Recommended):

Install in a Conda Environment (Recommended)
============================================

We highly recommend installing Vibrations in an isolated Conda environment.

The Conda environment can either be created manually (`Create Conda Environment`_), with the required
dependencies installed afterwards (`Installation Variants`_), or from one of the predefined environment
files provided in the ``requirements/`` directory (`Conda Environment from Requirements File`_).

The recommended way to install Conda is to follow the instructions on the
Conda homepage (https://conda.io/).

**Conda Initialization**

If Conda is installed but not yet initialized, initialize it for your shell.
For example, for ``zsh``:

.. code-block:: console

   % conda init zsh

Or Conda can be used without initialization by running:

.. code-block:: console

   % source activate
   (base) %

.. _Create Conda Environment:

**Create Conda Environment**

If Conda is already installed and initialized, create a new environment:

.. code-block:: console

   (base) % conda create --name VibENV

Activate the environment:

.. code-block:: console

   (base) % conda activate VibENV
   (VibENV) %

.. _Install Vibrations with pip:

Install Vibrations with pip
---------------------------

Vibrations uses the ``pyproject.toml`` file to define the package metadata and dependencies.

Before installing Vibrations, make sure that Python 3 and ``pip`` are available in the active environment.
If ``pip`` is not yet installed in the Conda environment, install it with:

.. code-block:: console
   
   (VibENV) % conda install pip

.. _Installation Variants:

**Installation Variants**

There are different predefined setup variants depending on your use case and preferred Open Babel package:

If you want to use the ``openbabel`` package, install with

.. code-block:: console

   (VibENV) Vibrations % pip install ".[openbabel]"

If you want to use the ``openbabel-wheel`` package, install with

.. code-block:: console

   (VibENV) Vibrations % pip install ".[openbabel-wheel]"

If you want to run the tests and/or build the documentation, it is recommended to do a full installation:

.. code-block:: console

   (VibENV) Vibrations % pip install ".[full-openbabel]"

or

.. code-block:: console

   (VibENV) Vibrations % pip install ".[full-openbabel-wheel]"

For development, Vibrations can be installed in editable mode:

.. code-block:: console

   (VibENV) Vibrations % pip install -e ".[full-openbabel]"

or

.. code-block:: console

   (VibENV) Vibrations % pip install -e ".[full-openbabel-wheel]"

Additional documentation packages can be installed if required:

.. code-block:: console

   (VibENV) Vibration % pip install docstr-coverage sphinx-mdinclude

You can build the Fortran libraries automatically with

.. code-block:: console

   (VibENV) Vibrations % pip install ".[fints]"
 
Verify the installation by running this test file which is located in ``src/Vibrations/tests/``:

.. code-block:: console

   (VibENV) Vibrations/src/Vibrations/tests % python tests.py
   
and/or by running ``pytest`` (see :ref:`Verify Installation <Verify Installation>`).
    
.. _Conda Environment from Requirements File:

Conda Environment from Requirements File
----------------------------------------

The repository provides predefined Conda environments in the ``requirements/`` directory.
These environments provide the dependencies required for a basic Vibrations installation
and a installation including testing and documentation support.

**Basic Conda Environment**

For creating a Conda environment containing the required dependencies for Vibrations,
use ``requirements/environment.yml``.

.. code-block:: console
   
   (base) % cd Vibrations/
   (base) Vibrations % conda env create -f requirements/environment.yml
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
   #     $ conda activate VibENV
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate

After activating the environment, Vibrations can be installed from the root directory
of the repository using ``pip``:

.. code-block:: console

   (base) Vibrations % conda activate VibENV
   (VibENV) Vibrations % pip install .
   Processing ./.
     Installing build dependencies ... done
     Getting requirements to build wheel ... done
     Preparing metadata (pyproject.toml) ... done
   .
   .
   .
   Successfully built vibrations
   Installing collected packages: vibrations
   Successfully installed vibrations-1.0

For available installation variants with Open Babel, see `Installation Variants`_.

Verify the installation by running this test file which is located in ``src/Vibrations/tests/``:

.. code-block:: console

   (VibENV) Vibrations/src/Vibrations/tests % python tests.py

**Testing and Documentation Environment**

The ``requirements/environment_tests_doc.yml`` file provides a Conda environment containing
the dependencies required for testing and documentation.

From the root directory of the Vibrations repository, create the environment:

.. code-block:: console

   (base) % cd Vibrations/
   (base) Vibrations % conda env create -f requirements/environment_tests_doc.yml
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
   #     $ conda activate VibENV
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate

After activating the environment, Vibrations can be installed from the root directory
of the repository using ``pip``:

.. code-block:: console

   (base) Vibrations % conda activate VibENV
   (VibENV) Vibrations % pip install .
   Processing ./.
     Installing build dependencies ... done
     Getting requirements to build wheel ... done
     Preparing metadata (pyproject.toml) ... done
   .
   .
   .
   Successfully built vibrations
   Installing collected packages: vibrations
   Successfully installed vibrations-1.0

For available installation variants with Open Babel, see `Installation Variants`_.

Verify the installation by running this test file which is located in ``src/Vibrations/tests/``

.. code-block:: console

   (VibENV) Vibrations/src/Vibrations/tests % python tests.py

and/or by running ``pytest`` (see :ref:`Verify Installation <Verify Installation>`).

The required testing dependencies, including ``pytest``, are already installed through the
provided ``requirements/environment_tests_doc.yml``.

**PyADF Environment Including Testing and Documentation**

For workflows involving `PyADF <https://github.com/chjacob-tubs/pyadf-releases>`_, additional dependencies are included in the
``requirements/environment_tests_doc_pyadf.yml`` file.

From the root directory of the Vibrations repository, create the environment:

.. code-block:: console

   (base) % cd Vibrations/
   (base) Vibrations % conda env create -f requirements/environment_tests_doc_pyadf.yml
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
   #     $ conda activate VibENV
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate

After activating the environment, Vibrations can be installed from the root directory
of the repository using ``pip``:

.. code-block:: console

   (base) Vibrations % conda activate VibENV
   (VibENV) Vibrations % pip install .
   Processing ./.
     Installing build dependencies ... done
     Getting requirements to build wheel ... done
     Preparing metadata (pyproject.toml) ... done
   .
   .
   .
   Successfully built vibrations
   Installing collected packages: vibrations
   Successfully installed vibrations-1.0

For available installation variants with Open Babel, see `Installation Variants`_.

Verify the installation by running this test file which is located in ``src/Vibrations/tests/``

.. code-block:: console

   (VibENV) Vibrations/src/Vibrations/tests % python tests.py

and/or by running ``pytest`` (see :ref:`Verify Installation <Verify Installation>`).

.. _Install in a Python Virtual Enviroment:

Install in a Python Virtual Environment
=======================================

Alternatively, a Python virtual environment can be created using the ``venv`` module:

.. code-block:: console

   python -m venv VibENV

Activate the virtual environment:

.. code-block:: console

   source VibENV/bin/activate

After activation, Vibrations can be installed using ``pip``.

For a basic installation:

.. code-block:: console 

   (VibENV) Vibrations % pip install ".[openbabel]"

For a full installation, including tests and documentation:

.. code-block:: console

   (VibENV) Vibrations % pip install ".[full-openbabel]"
   (VibENV) Vibrations % pip install ".[fints]"
   (VibENV) Vibrations % pip install docstr-coverage sphinx-mdinclude

For other installation variants, see `Installation Variants`_.

Verify the installation by running this test file which is located in ``src/Vibrations/tests/``

.. code-block:: console

   (VibENV) Vibrations/src/Vibrations/tests % python tests.py

and/or by running ``pytest`` (see :ref:`Verify Installation <Verify Installation>`).

.. _Manual Installation (PYTHONPATH):

Manual Installation (PYTHONPATH)
================================

The subdirectory ``Vibrations/src/Vibrations/`` has to be included in the
``PYTHONPATH`` environment variable.

The ``.zprofile`` file can be modified by adding:

.. code-block:: console

   export PYTHONPATH="${PYTHONPATH}:/path/to/Vibrations/src/"

Or the following lines can be added to a script before importing Vibrations:

.. code-block:: console

   import sys
   sys.path.append('/path/to/Vibrations/src/')


..
 .. _Verify Installation:
   
 *******************
 Verify Installation
 *******************

 Verify the installation by running ``tests.py`` which is located in ``src/Vibrations/tests/``:

 .. code-block:: console

    (VibENV) Vibrations/src/Vibrations/tests % python tests.py

 This works with all installation modes.

 If you have installed Vibrations with testing dependencies,
 you can also run the complete test suite using ``pytest``:

 .. code-block:: console

    (VibENV) Vibrations/src/Vibrations/unittest % pytest -v

 .. note::
   Another option to check the executability of the program is to calculate the code examples 
   located in ``examples/``.

   See also: :doc:`examples`.

..
 .. _Building the Documentation:

 **************************
 Building the Documentation
 **************************

 You can find the documentation, as mentioned above,
 on the homepage (https://vibrations.readthedocs.io/en/latest/),
 or generate the HTML documentation yourself with Sphinx.

 Build documentation via Sphinx and extension packages
 (only works with installations that include documentation dependencies):

 .. code-block:: console

    doc % sphinx-build . build

 Open the ``index.html`` file to view the documentation homepage:

 .. code-block:: console

    doc/build % open index.html

 .. note:: The documentation is automatically built and deployed as defined in ``.gitlab-ci.yml``.
