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
 The prerequisite for the check is that we have pytest installed.

 Go to the appropiate test folder:

    >>> % cd vibrations/unittests/

 Run the test:

    >>> vibrations/unittests/% pytest -v -p no:warnings

 If everything runs correctly, we will get the following output:

 .. code-block:: console

    ================================ test session starts ======================================
    platform linux -- Python 3.11.8, pytest-8.0.2, pluggy-1.4.0 -- /home/usr/.conda/envs/VibENV
    /bin/python3.11
    cachedir: .pytest_cache
    rootdir: /home/usr/sources/vibrations
    collected 113 items                                                                             
   
    test_Grids.py::test_localize_subsets PASSED                                        [  0%]
    test_Grids.py::test_Grid_class_init_empty PASSED                                   [  1%]
    test_Grids.py::test_Grid_class_init PASSED                                         [  2%]
    test_Grids.py::test_Grid_generate_grids PASSED                                     [  3%]
    test_Grids.py::test_Grid_get_grid_structure PASSED                                 [  4%]
    test_Grids.py::test_Grid_get_molecule PASSED                                       [  5%]
    test_Grids.py::test_Grid_get_pyadf_molecule PASSED                                 [  6%]
    .
    .
    .
    test_VSCF.py::test_VSCF2D_save_wave_functions PASSED                               [ 95%]
    test_VSCF.py::test_VSCF2D_print_results PASSED                                     [ 96%]
    test_Wavefunction.py::test_Wavefunction_init_empty PASSED                          [ 97%]
    test_Wavefunction.py::test_Wavefunction_init PASSED                                [ 98%]
    test_Wavefunction.py::test_Wavefunction_save PASSED                                [ 99%]
    test_Wavefunction.py::test_Wavefunction_read PASSED                                [100%]
   
    =============================== 113 passed in 66.52s (0:01:06) ===========================


 .. note::
    Another possibility to check the executability of the program is to calculate the code examples. 
    See :doc:`Application Examples <examples>`
