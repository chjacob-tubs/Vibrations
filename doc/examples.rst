########
Examples
########

See the ``/LocVib/src/VibTools/example/`` directory for some examples of typical runs.


.. code-block:: console

   /LocVib/src/VibTools/example/
   ├── 1_composition.py
   ├── 2_locmodes.py
   ├── 3_couplings.py
   └── Ala10
       ├── coord
       ├── restart
       └── snf.out

As a typical example of a vibrational spectra calculation using localized modes, we show the results of an **SNF** calculation (https://reiher.ethz.ch/software/snf.html) for an *Ala10* molecule (harmonic approximation).

composition
===========

The ``1_composition.py`` script provides an example of calculating vibrational compositions:

.. dropdown:: Open Python Script - 1_composition.py

   .. literalinclude:: ../src/VibTools/example/1_composition.py
       :language: python

Running the script, generates the following output:

   >>> /LocVib/src/VibTools/example % python 1_composition.py

.. dropdown:: Open Output: python 1_composition.py

   .. literalinclude:: examples_output/1_output.txt
       :language: text

locmodes
========

The ``2_locmodes.py`` script provides an example of localizing normal modes:

.. dropdown:: Open Python Script - 2_locmodes.py

   .. literalinclude:: ../src/VibTools/example/2_locmodes.py
       :language: python

Running the script, generates the following output:

   >>> /LocVib/src/VibTools/example % python 2_locmodes.py

.. dropdown:: Open Output: python 2_locmodes.py

   .. literalinclude:: examples_output/2_output.txt
       :language: text


couplings
=========

The ``3_couplings.py`` script provides an example of calculating couplings between localized modes:

.. dropdown:: Open Python Script - 3_couplings.py

   .. literalinclude:: ../src/VibTools/example/3_couplings.py
       :language: python

Running the script, generates the following output:

   >>> /LocVib/src/VibTools/example % python 3_couplings.py

.. dropdown:: Open Output: python 3_couplings.py

   .. literalinclude:: examples_output/3_output.txt
       :language: text
