Contrail Detection Algorithms
=============================

Contrail detection algorithm objects are constructed to take an imager and a
scene and return a binary mask indicating the presence of contrails in the scene.

In applying the algorithm, the imager is used to simulate the observation of the
scene—the fields don't need pre-calculating.

.. autoclass:: simtrails.detection_algorithms.ContrailDetectionAlgorithm
    :members:

.. autoclass:: simtrails.detection_algorithms.MannsteinCDA
    :members:
