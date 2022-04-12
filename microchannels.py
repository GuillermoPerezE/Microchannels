import math
import numpy as np
import matplotlib.pyplot as plt
import materials


class Microchannel():
    W_d = 51e-3  # Heat sink's width (m)
    L_d = 51e-3  # Heat sink's length (m)
    H_c = 1.7e-3  # Channel's height (m)
    H_b = 0.1e-3  # Channel's base (m)
    H_d = H_c + H_b  # Heat sink height (m)
    W_i = 51e-3  # Interface's width (m)
    L_i = 51e-3  # Interface's length (m)

    T_a = 300  # Ambient temperature (K)
    R_i_by_A = 2.75e-4  # (K.m2/W)

    L_tu = 0.5  # Supply tube length (m)
    D_tu = 1e-2  # Supply tube diameter (m)

    def __init__(self, base=materials.Copper(), coolant=materials.Water()):
        # Read materials
        self._bmat = base
        self._cmat = coolant

        # Fluid flow rate (m3/s)
        self.G_d = 7e-3

        # Heat flux density (W/m2)
        self.q = 5e4

        # Geometry properties
        self.w_c = (250e-6 / 2)
        self.w_w = (140e-6 / 2)

        # self.gamma = 1.4
        # self.Rgas = 287.0028305

    @property
    def Pr(self):
        # Prandtl number
        return self._cmat.nu * self._cmat.rho * self._cmat.c_p / self._cmat.k

    @property
    def N(self):
        # Number of channels
        return (self.W_d / 2 - self.w_w) // (self.w_c + self.w_w)

    @property
    def alpha(self):
        # Channel's aspect ratio
        return 2 * self.w_c / self.H_c

    @property
    def beta(self):
        # Channel-wall ratio
        return self.w_c / self.w_w

    @property
    def A_b(self):
        return self.W_d * self.L_d

    @property
    def A_i(self):
        return self.W_i * self.L_i

    @property
    def Q(self):
        return self.q * self.A_i

    # %% Thermal resistance calculations
    @property
    def R_f(self):
        # Fluid thermal resistance [K/W]
        return 1.0 / (self._cmat.rho * self.G_d * self._cmat.c_p)

    @property
    def T_f(self):
        # Fluid temperature [K]
        return self.T_a + self.R_f * self.Q

    @property
    def h_avg(self):
        # Film (convection) coefficient [W/K.m2]
        return self._cmat.k * self.Nu_Dh / self.D_h

    @property
    def eta_p(self):
        # Fin efficiency
        mHc = np.sqrt(2.0 * (2.0 * self.w_w + self.L_d) * self.h_avg / (
                self._bmat.k * 2.0 * self.w_w * self.L_d)) * self.H_c
        return np.tanh(mHc) / mHc

    @property
    def R_conv(self):
        # Convection thermal resistance [K/W]
        A_eff = 2.0 * self.N * (self.eta_p * self.H_c + self.w_c) * self.L_d
        return 1.0 / (self.h_avg * A_eff)

    @property
    def T_s(self):
        return self.T_f + self.R_conv * self.Q

    @property
    def R_o(self):
        # Thermal resistance of the microchannel heat sink [K/W]
        return self.R_conv + self.R_f

    @property
    def Bi(self):
        # Biot's number
        return 1.0 / (np.pi * self._bmat.k * self.R_o)

    @property
    def R_b(self):
        # Thermal resistance due contraction/dispersion
        sqrtpi = np.sqrt(np.pi)
        a = np.sqrt(self.A_i / np.pi)
        b = np.sqrt(self.A_b / np.pi)
        tau = self.H_b / b
        epsilon = a / b
        lambda_c = np.pi + 1.0 / (sqrtpi * epsilon)
        Phi_c = (np.tanh(lambda_c * tau) + lambda_c / self.Bi ) / (
                1.0 + lambda_c * np.tanh(lambda_c * tau) / self.Bi)
        Psi_avg = epsilon * tau / sqrtpi + 0.5 * Phi_c * \
                  (1.0 - epsilon) ** (3 / 2)
        return Psi_avg / (sqrtpi * self._bmat.k * a)

    @property
    def T_b(self):
        return self.T_s + self.R_b * self.Q

    @property
    def R_i(self):
        # Thermal interface resistance [K/W]
        return self.R_i_by_A / self.A_i

    @property
    def T_i(self):
        # Interfacial temperature [K]
        return self.T_b + self.R_i * self.Q

    @property
    def R_eq(self):
        # Equivalent thermal resistance [K/W]
        return self.R_o + self.R_b + self.R_i

    # %% Sgen_ff calculations
    @property
    def DeltaP_mh(self):
        # Pressure Drop inside microchannels
        k_ce = 1.79 - 2.32 * (self.beta / (1 + self.beta)) + \
               0.53 * (self.beta / (1 + self.beta)) ** 2
        return 0.5 * self._cmat.rho * self.U_avg ** 2 * (
                self.f * (self.L_d / self.D_h) + k_ce)

    @property
    def U_avg_tu(self):
        # Average fluid velocity inside the supply tube
        return 4.0 * self.G_d / (np.pi * self.D_tu ** 2)

    @property
    def Re_D_tu(self):
        # Reynolds' number inside the supply tube
        return self.U_avg_tu * self.D_tu / self._cmat.nu

    @property
    def f_tu(self):
        # Friction factor inside the supply tube
        return 4.0 * (0.09290 + 1.01612 / (self.L_tu / self.D_tu)) * \
               self.Re_D_tu ** (-0.268 - 0.3193 / (self.L_tu / self.D_tu))

    @property
    def A_tu_hs(self):
        # Ratio between areas of tube's transversal section and heat sink
        return 0.25 * np.pi * self.D_tu ** 2 / (self.W_d * self.H_c)

    @property
    def DeltaP_tu(self):
        # Pressure drop inside the supply tube
        return 0.5 * self._cmat.rho * self.U_avg_tu ** 2 * (
                0.42 + (1 - self.A_tu_hs ** 2) ** 2 + 0.42 * (
                1 - self.A_tu_hs ** 2) + 1. + 2 * self.f_tu * self.L_tu / self.D_tu)

    @property
    def DeltaP_total(self):
        # Total Pressure Drop
        return self.DeltaP_mh + self.DeltaP_tu

    @property
    def Phi(self):
        # Pumping power
        return self.G_d * self.DeltaP_total

    # %% Entropy-related calculations
    @property
    def sgen_ht(self):
        return self.Q ** 2 * self.R_eq / (self.T_a * self.T_i)

    @property
    def sgen_ff(self):
        return self.G_d * self.DeltaP_total / self.T_a

    @property
    def sgen(self):
        return self.sgen_ht + self.sgen_ff

    @property
    def Be(self):
        # Bejan's number
        return self.sgen_ht / self.sgen

    # %%
    # @property
    # def U_avg_sound(self):
    #     return (self.gamma * self.Rgas * self.T_a) ** (1 / 2)

    @property
    def D_h(self):
        # Hydraulic diameter
        ac = 2 * self.w_c * self.H_c
        p = 2 * (self.H_c + 2 * self.w_c)
        return 4 * ac / p

    @property
    def mdot(self):
        # Mass flow
        return self._cmat.rho * self.G_d / (2 * self.N)

    @property
    def U_avg(self):
        # Mass velocity
        return self.mdot / (self._cmat.rho * self.w_c * self.H_c)

    @property
    def Ma(self):
        # Mach number
        return self.U_avg / self.U_avg_sound

    @property
    def flow_regime(self):
        return 'laminar' if self.Re_D_tu < 2300 else 'turbulent'

    @property
    # Friction coefficient
    def f(self):
        if self.flow_regime == 'laminar':
            ab_constant = (self.alpha ** 2 + 1) / ((self.alpha + 1) ** 2)
            return ((3.2 * (self.Re_D_tu * self.D_h / self.L_d) ** 0.57) ** 2 + (
                        4.7 + 19.64 * ab_constant) ** 2) ** (1 / 2) / self.Re_D_tu
        else:
            return 1. / (0.79 * np.log(self.Re_D_tu) - 1.64) ** 2

    @property
    # Nusselt number
    def Nu_Dh(self):
        if self.flow_regime == 'laminar':
            return 2.253 + 8.164 * (1 / (self.alpha + 1)) ** 1.5
        else:
            return (self.f / 2) * (self.Re_D_tu - 1000) * self.Pr / (
                        1 + 12.7 * np.sqrt(self.f / 2) * (self.Pr ** (2 / 3) - 1))

    def plotc(self):
        z = 0
        sgenp = np.arange(10000)
        wcp = np.arange(10000)
        wwp = np.arange(10000)
        sgenp = sgenp.astype(float)
        wcp = wcp.astype(float)
        wwp = wwp.astype(float)
        for x in range(100):
            self.w_c = (250e-6 / 2) - (x * 1e-6)
            self.w_w = (250e-6 / 2)
            for y in range(100):
                self.w_w = (250e-6 / 2) - (y * 1e-6)
                sgenp[z] = self.sgen
                wcp[z] = self.w_c
                wwp[z] = self.w_w
                z = z + 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.xlabel('Wc')
        plt.ylabel('Ww')
        ax.set_zlabel(r'$\dot{S}_{gen}$')
        ax.scatter(wcp, wwp, sgenp)

        plt.show()

        # wc = np.linspace(125e-6, 0.0004, 100)
        # sgenp = []
        #wc = []

        # for wc_i in wc:
        #     self.wc = wc_i
        #     sgenp.append(self.tegr())

        # # ax = plt.axes(projection='3d')
        # plt.xlabel(r'$w_c$')
        # # plt.ylabel(r'$\beta$')
        # plt.ylabel(r'$\dot{S}_{gen}$')
        # plt.plot(np.array(wc), np.array(sgenp))
        # plt.show()


if __name__ == '__main__':
    mc = Microchannel()
    mc.plotc()
