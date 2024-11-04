Detectors and Sensitivity Tests
===============================

Detectors
---------

Detectors combine contrail detection algorithms with scenes and imagers to 
simulate the detectability of a contrail.

.. automodule:: simtrails.contrail_detector
    :members:

Sensitivity tests
-----------------

Sensitivity tests are used to determine the sensitivity of contrail detection
to variations in imager or scene parameters.

The sensitivity test uses ``detector_generator`` and ``detectable_generator``,
to create instances of detectors and detectables with varying parameters.
These can be simply copies of the original detector/detectable, or can be
more complex instances of ``InstanceGenerator``.

.. automodule:: simtrails.sensitivity_test
    :members:

