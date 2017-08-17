"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""


class Material:
    """Material.

    Class used to create a material and define its properties
    """
    def __init__(self, E, G_s, rho):
        self.E = E
        self.G_s = G_s
        self.rho = rho
        self.color = None  # this can be used in the plots


