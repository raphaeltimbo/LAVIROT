"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""


class Material:
    """Material.

    Class used to create a material and define its properties

    Attributes
    ----------

    E : float
        Young's modulus.
    G_s : float
        Shear modulus.
    rho : float
        Density.
    """
    def __init__(self, rho=None, E=None, G_s=None, Poisson=None):
        self.rho = rho
        self.E = E
        self.G_s = G_s
        self.Poisson = Poisson
        if G_s is None:
            self.G_s = E/(2*(1 + Poisson))
        elif Poisson is None:
            self.Poisson = (E/(2*G_s)) - 1
        # TODO If E is None
        # TODO Implement tests

        self.color = None  # this can be used in the plots

