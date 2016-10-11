import numpy as np

class Rotor(object):
    """A rotor object.

    This class will create a rotor with the shaft,
    disk and bearing elements provided.

    Parameters
    ----------
    shaft_elements: list
        List with the shaft elements
    bearing_elements: list
        List with the bearing elements
    disk_elements: list
        List with the disk elements

    Returns
    ----------
    A rotor object.

    Examples:

    """

    def __init__(self, shaft_elements, bearing_elements, disk_elements):
        self.shaft_elements = shaft_elements
        self.bearing_elements = bearing_elements
        self.disk_elements = disk_elements
        #  number of dofs
        self.ndof = 4 * len(shaft_elements) + 4

        #  ========== Assembly the rotor matrices ==========

        #  Create the matrices
        M0 = np.zeros((self.ndof, self.ndof))
        C0 = np.zeros((self.ndof, self.ndof))
        G0 = np.zeros((self.ndof, self.ndof))
        K0 = np.zeros((self.ndof, self.ndof))

        #  Skew-symmetric speed dependent contribution to element stiffness matrix
        #  from the internal damping.
        K1 = np.zeros((self.ndof, self.ndof))

        #  Shaft elements
        for elm in shaft_elements:
            node = elm.n
            n1 = 4 * node
            n2 = 4 * node + 8

            M0[n1:n2, n1:n2] += elm.M()

        self.M = M0




