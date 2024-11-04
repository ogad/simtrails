.. Simtrails documentation master file, created by
   sphinx-quickstart on Fri Nov  1 10:37:15 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simtrails documentation
=======================

Simtrails is a package for simulating synthetic satellite images of contrails
using radiative transfer modelling (via `libRadtran <https://www.libradtran.org/>`_
and the `pyLRT <https://github.com/EdGrrr/pyLRT/>`_ python wrapper), and test the ability
of contrail detection algorithms to detect the simulated contrails.

Details of contrail detectability are found in the paper by Driver et al. (2024) [1]_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/installation
   detectables
   imagers
   detectors
   sensitivity_result
   radiative_transfer
   detection_algorithms
   mannstein
   cocip
   misc
   


References
==========
.. [1] Driver, O. G. A., Stettler, M. E. J., and Gryspeerdt, E.: Factors limiting contrail detection in satellite imagery, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-2198, 2024.d