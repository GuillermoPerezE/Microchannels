class Material:
    model_kind = 'simple'

    def __init__(self, phase=None, **ambient_parameters):
        # Phase of the material
        self.phase = phase

        # Thermal conductivity (W/m.K)
        self.k = None

        # Density (kg/m3)
        self.rho = None

        # Kinematic viscosity (m2/s)
        self.nu = None

        # Specific heat (J/kg.K)
        self.c_p = None


# %% Solid materials
class Aluminum(Material):
    """
    This is Aluminum (Al) material.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Solid', **ambient_parameters)

        self.k = 237
        self.rho = 2707


class Copper(Material):
    """
    This is Copper (Cu) material.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Solid', **ambient_parameters)

        self.k = 401
        self.rho = 8954


# %% Fluid materials
class Air(Material):
    """
    This is Air fluid.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Fluid', **ambient_parameters)

        self.k = 0.0261
        self.rho = 1.1614
        self.nu = 1.58e-5
        self.c_p = 1007


class Water(Material):
    """
    This is Water (H2O) fluid.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Fluid', **ambient_parameters)

        self.k = 0.625
        self.rho = 994.2
        self.nu = 7.25e-4/self.rho
        self.c_p = 4178
