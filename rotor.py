import numpy as np

class Rotor(object):
    """A rotor object.

    This class will create a rotor with the shaft,
    disk and bearing elements provided.

    Parameters
    ----------
    shaft_elements: list
        List with the shaft elements
    disk_elements: list
        List with the disk elements
    bearing_elements: list
        List with the bearing elements

    Returns
    ----------
    A rotor object.

    Examples:

    """

    def __init__(self, shaft_elements, disk_elements, bearing_elements):
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
        #  TODO add the contribution for K1 matrix
        K1 = np.zeros((self.ndof, self.ndof))

        #  Shaft elements
        for elm in shaft_elements:
            node = elm.n
            n1 = 4 * node   # first dof
            n2 = n1 + 8     # last dof

            M0[n1:n2, n1:n2] += elm.M()
            G0[n1:n2, n1:n2] += elm.G()
            K0[n1:n2, n1:n2] += elm.K()

        #  TODO Add error for elements with the same n (node)

        #  Disk elements
        for elm in disk_elements:
            node = elm.n
            n1 = 4 * node
            n2 = n1 + 4     # Disk is inserted in the dofs (4) of the first node

            M0[n1:n2, n1:n2] += elm.M()
            G0[n1:n2, n1:n2] += elm.G()

        self.M = M0
        self.C = C0
        self.G = G0
        self.K = K0
