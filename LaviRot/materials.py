"""Materials module.

This module defines the Material class and defines
some of the most common materials used in rotors.
"""
import numpy as np

__all__ = ['Material', 'available_materials',
           'Oil']


class Material:
    """Material.

    Class used to create a material and define its properties.
    Density and at least at least 2 arguments from E, G_s and
    Poisson should be provided.

    See available_materials for materials already provided.

    Parameters
    ----------
    name : str
        Material name.
    E : float
        Young's modulus.
    G_s : float
        Shear modulus.
    rho : float
        Density.

    Examples
    --------
    >>> AISI4140 = Material(name='AISI4140', rho=7850, E=203.2e9, G_s=80e9)
    >>> AISI4140.Poisson
    0.27

    """
    material_instances = []

    def __init__(self, name=None, rho=None, E=None, G_s=None, Poisson=None,
                 color='#525252'):
        if rho is None:
            raise ValueError('Density (rho) not provided.')

        none_args = []
        for arg in ['E', 'G_s', 'Poisson']:
            if locals()[arg] is None:
                none_args.append(arg)
        if len(none_args) > 1:
            raise ValueError('At least 2 arguments from E, G_s'
                             'and Poisson should be provided ')

        self.name = name
        self.rho = rho
        self.E = E
        self.G_s = G_s
        self.Poisson = Poisson
        if E is None:
            self.E = G_s*(2*(1 + Poisson))
        elif G_s is None:
            self.G_s = E/(2*(1 + Poisson))
        elif Poisson is None:
            self.Poisson = (E/(2*G_s)) - 1

        self.color = color  # this can be used in the plots

        Material.material_instances.append(name)

    def __repr__(self):
        return f'{self.name}'

    def __str__(self):
        return (
            f'{self.name}'
            f'\n{35*"-"}'
            f'\nDensity         (N/m**3): {float(self.rho):{2}.{8}}'
            f'\nYoung`s modulus (N/m**2): {float(self.E):{2}.{8}}'
            f'\nShear modulus   (N/m**2): {float(self.G_s):{2}.{8}}'
            f'\nPoisson coefficient     : {float(self.Poisson):{2}.{8}}')


#####################################################################
# Available materials
#####################################################################

steel = Material(name='Steel', rho=7810, E=211e9, G_s=81.2e9)
AISI4140 = Material(name='AISI4140', rho=7850, E=203.2e9, G_s=80e9)

available_materials = Material.material_instances


class Oil:
    """Oil.

    Class used to create an oil and define its properties.

    See available_oils for oils already provided.

    Parameters
    ----------
    name : str
        Oil name.
    exp_coeff : float
        Thermal expansion coefficient. Defaults to 0.00076
    t_a : float
        Temperature at point a.
    mu_a : float
        Viscosity at point a.
    rho_a : float
        Density at point a.
    t_b : float
        Temperature at point b.
    mu_b : float
        Viscosity at point b.


    Examples
    --------
    """
    def __init__(self, name=None, exp_coeff=0.00076,
                 t_a=None, mu_a=None, rho_a=None,
                 t_b=None, mu_b=None):

        self.name = name
        self.exp_coeff = exp_coeff

        self.t_a = t_a
        self.mu_a = mu_a
        self.rho_a = rho_a
        self.v_a = mu_a / rho_a

        self.t_b = t_b
        self.mu_b = mu_b
        self.rho_b = self.rho(t_b)
        self.v_b = mu_b / self.rho_b

    def rho(self, T):
        """Density."""
        return self.rho_a * (1 - self.exp_coeff * (T - self.t_a))

    def v(self, T):
        """Kinematic viscosity."""
        va = self.v_a
        vb = self.v_b
        ta = self.t_a
        tb = self.t_b

        v = va * np.exp((np.log(vb/va)*(T - ta))/(tb - ta))

        return v

