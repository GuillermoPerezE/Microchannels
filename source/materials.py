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

class Aluminium(Material):
    """
    This is Aluminium (Al) material.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Solid', **ambient_parameters)

        self.k = 237
        self.rho = 2707

class Titanium(Material):
    """
    This is Titanium (Ti) material.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Solid', **ambient_parameters)

        self.k = 22.4
        self.rho = 4507

class Graphite(Material):
    """
    This is High Thermal Conductive Graphite (HTCG) material.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Solid', **ambient_parameters)

        self.k = 1900
        self.rho = 1000

class Silicon(Material):
    """
    This is Silicon(Si) material.
    """
    def __init__(self, **ambient_parameters):
        super().__init__('Solid', **ambient_parameters)

        self.k = 148
        self.rho = 2330

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

# %% Nanofluids

class AlNF(Material):
    """
    This is H20-Al Nanofluid at a volume concentration of 0.05 .
    """
    def __init__(self, concent=0.001, **ambient_parameters):
        super().__init__('Fluid', **ambient_parameters)

        #self.k = 0.625*((237+(6-1)*0.625+(6-1)*0.05*(237-0.625))/(237+(6-1)*0.625-0.05*(237-0.625)))
        self.rho = concent * 2707 + (1 - concent) * 994.2
        self.k = 0.625*((237 + 2 * 0.625 - 2 * self.rho * (0.625-237)) / (237 + 2 * 0.625 + concent * (0.625 - 237)))
        self.nu = (7.25e-4 / 994.2) * (1 + 2.5 * concent)
        self.c_p = (concent * 2707 * 910 + (1 - concent) * 994.2 * 4178) / self.rho

class TiONF(Material):
    """
    This is H20-TiO2 Nanofluid at a volume concentration of 0.05 .
    """
    def __init__(self, concent=0.001, **ambient_parameters):
        super().__init__('Fluid', **ambient_parameters)

        #self.k = 0.625*((8.4+(6-1)*0.625+(6-1)*0.05*(8.4-0.625))/(8.4+(6-1)*0.625-0.05*(8.4-0.625)))
        self.rho = concent * 4157 + (1 - concent) * 994.2
        self.k = 0.625 * ((8.4 + 2 * 0.625 - 2 * self.rho * (0.625 - 8.4)) / (8.4 + 2 * 0.625 + concent * (0.625 - 8.4)))
        self.nu = (7.25e-4 / 994.2) * (1 + 2.5 * concent)
        self.c_p = (concent * 4157 * 710 + (1 - concent) * 994.2 * 4178) / self.rho