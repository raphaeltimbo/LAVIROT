
Compressor
==========

.. code:: ipython3

    import LaviRot as lr
    import numpy as np
    %matplotlib inline

For this example we will consider a compressor with shaft and impellers
made from AISI 4140, which has the following properties.

.. code:: ipython3

    material = lr.materials.AISI4140

Shaft's inner and outer diameter will be:

.. code:: ipython3

    si_d = 0
    so_d = 0.127

Now we define the length for each element, starting at the first element
on the left (the element that will have the thrust collar).

.. code:: ipython3

    L = [0.07, # thrust collar
        0.058,
        0.058, # 0 probe
        0.052, # 0 bearing
        0.092,
        0.092,
        0.092,
        0.092, # 0 impeller
        0.082, # 1 impeller
        0.082, # 2 impeller
        0.082, # 3 impeller
        0.082, # 4 impeller
        0.082, # 5 impeller
        0.068, # honeycomb # change diameter
        0.086,
        0.086,
        0.086,
        0.086, # 1 bearing
        0.086,
        0.086]

The next line defines a list with the number of each element (from 0 to
len(L)).

.. code:: ipython3

    nelem = [x for x in range(len(L))]

Now we create a list with all the shaft elements using the :ref:`ShaftElement` class. In this case we are going to consider Timoshenko beam elements (with shear and rotary inertia effects).

.. code:: ipython3

    shaft_elem = [lr.ShaftElement(l, si_d, so_d, material,
                                  shear_effects=True,
                                  rotary_inertia=True,
                                  gyroscopic=True) for l in L]

The disks are created using the :ref:`DiskElement` class.

.. code:: ipython3

    colar = lr.DiskElement(1, material, 0.035, so_d, 0.245)
    disk0 = lr.DiskElement(8, material, 0.02, so_d, 0.318)
    disk1 = lr.DiskElement(9, material, 0.02, so_d, 0.318)
    disk2 = lr.DiskElement(10, material, 0.02, so_d, 0.318)
    disk3 = lr.DiskElement(11, material, 0.02, so_d, 0.318)
    disk4 = lr.DiskElement(12, material, 0.02, so_d, 0.318)
    disk5 = lr.DiskElement(13, material, 0.02, so_d, 0.318)

For the bearings we use the :ref:`BearingElement` class. We will consider a constant stifness for the bearings.

.. code:: ipython3

    stfx = 1e8
    stfy = 1e8
    bearing0 = lr.BearingElement(4, kxx=stfx, kyy=stfy, cxx=0, cyy=0)
    bearing1 = lr.BearingElement(-3, kxx=stfx, kyy=stfy, cxx=0, cyy=0)

Now we assemble the compressor rotor using the :ref:`Rotor` class.

.. code:: ipython3

    compressor = lr.Rotor(shaft_elem,
                          [colar, disk0, disk1, disk2, disk3, disk4, disk5],
                          [bearing0, bearing1])

We can now use the function method plot_rotor.

.. code:: ipython3

    compressor.plot_rotor()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x7f67f17bfa58>




.. image:: compressor_files/compressor_19_1.png


Now we are going to check the natural frequencies using the Campbell
diagram. First we need to define the speed range that we want to
analyze.

.. code:: ipython3

    speed = np.linspace(0, 1500, 10)

Now we can call the campbell method with mult=[1, 2] to plot 1x and 2x the speed.

.. code:: ipython3

    compressor.campbell(speed, mult=[1, 2])




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x7f67edee4c88>




.. image:: compressor_files/compressor_23_1.png


