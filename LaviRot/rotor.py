import numpy as np
import scipy.linalg as la


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
        #  TODO consider speed as a rotor property. Setter should call __init__ again
        self.shaft_elements = shaft_elements
        self.bearing_elements = bearing_elements
        self.disk_elements = disk_elements
        #  TODO check when disk diameter in no consistent with shaft diameter
        #  TODO add error for elements added to the same n (node)
        #  number of dofs
        self.ndof = 4 * len(shaft_elements) + 4

        #  nodes axial position
        nodes_pos = [s.z for s in self.shaft_elements]
        # append position for last node
        nodes_pos.append(self.shaft_elements[-1].z
                         + self.shaft_elements[-1].L)
        self.nodes_pos = nodes_pos

        #  TODO for tappered elements i_d and o_d will be a list with two elements
        #  diameter at node position
        nodes_i_d = [s.i_d for s in self.shaft_elements]
        # append i_d for last node
        nodes_i_d.append(self.shaft_elements[-1].i_d)
        self.nodes_i_d = nodes_i_d

        nodes_o_d = [s.o_d for s in self.shaft_elements]
        # append o_d for last node
        nodes_o_d.append(self.shaft_elements[-1].o_d)
        self.nodes_o_d = nodes_o_d

    @staticmethod
    def dofs(element):
        """This function will return the first and last dof
        for a given element"""
        if type(element).__name__ == 'ShaftElement':
            node = element.n
            n1 = 4 * node
            n2 = n1 + 8
        if type(element).__name__ == 'DiskElement':
            node = element.n
            n1 = 4 * node
            n2 = n1 + 4
        if type(element).__name__ == 'BearingElement':
            node = element.n
            n1 = 4 * node
            n2 = n1 + 2
        # TODO implement this for bearing with more dofs
        return n1, n2

    def M(self):
        """This method returns the rotor mass matrix"""
        #  Create the matrices
        M0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self.dofs(elm)
            M0[n1:n2, n1:n2] += elm.M()

        for elm in self.disk_elements:
            n1, n2 = self.dofs(elm)
            M0[n1:n2, n1:n2] += elm.M()

        return M0

    def K(self):
        """This method returns the rotor stiffness matrix"""
        #  Create the matrices
        K0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self.dofs(elm)
            K0[n1:n2, n1:n2] += elm.K()

        for elm in self.bearing_elements:
            n1, n2 = self.dofs(elm)
            K0[n1:n2, n1:n2] += elm.K()
        #  Skew-symmetric speed dependent contribution to element stiffness matrix
        #  from the internal damping.
        #  TODO add the contribution for K1 matrix

        return K0

    def C(self):
        """This method returns the rotor stiffness matrix"""
        #  Create the matrices
        C0 = np.zeros((self.ndof, self.ndof))

        for elm in self.bearing_elements:
            n1, n2 = self.dofs(elm)
            C0[n1:n2, n1:n2] += elm.C()

        return C0

    def G(self):
        """This method returns the rotor stiffness matrix"""
        #  Create the matrices
        G0 = np.zeros((self.ndof, self.ndof))

        for elm in self.shaft_elements:
            n1, n2 = self.dofs(elm)
            G0[n1:n2, n1:n2] += elm.G()

        for elm in self.disk_elements:
            n1, n2 = self.dofs(elm)
            G0[n1:n2, n1:n2] += elm.G()

        return G0

    def A(self, w=0):
        """This method creates a speed dependent space state matrix"""
        Z = np.zeros((self.ndof, self.ndof))
        I = np.eye(self.ndof)
        Minv = la.pinv(self.M())
        #  TODO implement K(w) and C(w) for shaft, bearings etc.
        A = np.vstack([np.hstack([Z, I]),
                       np.hstack([-Minv @ self.K(), -Minv @ (self.C() + self.G()*w)])])

        return A

    @staticmethod
    def index(eigenvalues):
        """
        Function used to generate an index that will sort
        eigenvalues and eigenvectors.
        """
        # positive in increasing order
        idxp = eigenvalues.imag.argsort()[int(len(eigenvalues)/2):]
        # negative in decreasing order
        idxn = eigenvalues.imag.argsort()[int(len(eigenvalues)/2) - 1::-1]

        idx = np.hstack([idxp, idxn])

        #  TODO implement sort that considers the cross of eigenvalues
        return idx

    def eigen(self, w=0, sorted_=True):
        """
        This method will return the eigenvalues and eigenvectors of the
        state space matrix A sorted by the index method.
        To avoid sorting use sorted_=False
        """
        evalues, evectors = la.eig(self.A(w))
        if sorted_ is False:
            return evalues, evectors

        idx = self.index(evalues)

        return evalues[idx], evectors[:, idx]
