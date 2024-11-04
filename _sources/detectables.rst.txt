Detectables
===========

"Detectables" describe scenes whose detection can be simulated by a detector.

They contain a "grid",  whose values are used by the object to determine relevant
parameters for radiative transfer calculations. For example, the grid values in 
contrail objects are used to scale the contrail's IWP. Other physical properties 
are fixed for the whole scene (and are stored as attributes of the object).

Note that RT calculations also need information about the relevant wavelengths,
so the object can't calculate the radiative transfer itself.

In normal usage, an instance of a detectable object is created, and then passed
to an imager object to simulate a synthetic observation of the scene, or passed 
to a detector object to simulate an attempt detection using the given 
imager/detction-algorithm/scene combination.


.. autoclass:: simtrails.detectable.Detectable
    :members:

Detectable contrail scenes
--------------------------

.. autoclass:: simtrails.contrail.Contrail
    :members:

.. autoclass:: simtrails.contrail.GaussianContrail
    :members: