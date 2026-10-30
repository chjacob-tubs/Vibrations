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
