import math
import numpy as np
import matplotlib.pyplot as plt
import materials


# Guillermo, ver este video antes de continuar: https://www.youtube.com/watch?v=jCzT9XFZ5bw


class Microchannel():
    # hidden properties
    # __Dh = 0
    # __Ab = 0
    # __Ai = 0
    # __f = 0
    # __NuDh = 0
    # __havg = 0
    # __etap = 0
    # __csound = 0
    # __mdot = 0
    # __Ma = 0
    # __Q = 0
    # __Ri = 0
    # __Rconv = 0
    # __Rf = 0
    # __Rb = 0
    # __Tf = 0
    # __Ts = 0
    # __Tb = 0
    # __Ti = 0
    # __Sgenht = 0
    # __Sgenff = 0
    # __Omega = 0
    # __Be = 0

    W_d = 51e-3  # Heat sink's width (m)
    L_d = 51e-3  # Heat sink's length (m)
    H_c = 1.7e-3  # Channel's height (m)
    H_b = 0.1e-3  # Channel's base (m)
    H_d = H_c + H_b  # Heat sink height (m)
    W_i = 51e-3  # Interface's width (m)
    L_i = 51e-3  # Interface's length (m)
    T_a = 300  # Ambient temperature (K)

    def __init__(self, base=materials.Copper(), coolant=materials.Water()):
        # Read materials
        self._bmat = base
        self._cmat = coolant
        # Geometry properties
        self.w_c = (250e-6 / 2)
        self.w_w = (140e-6 / 2)

        # Fluid properties
        self.Ta = 300
        self.G = 7e-3
        self.nu = 1.58e-5
        self.cp = 1007
        # self.Pr = .71
        self.gamma = 1.4
        self.Rgas = 287.0028305
        self.rho = 1.1614
        self.kf = 0.0261
        self.Uavg = 0
        self.Regime = 'laminar'
        self.ReDh = 0

        # Material properties
        self.RicondA = 2.75e-4
        self.k = 148
        self.q = 15e4
        # Performance parameters
        self.Req = 0
        self.DeltaP = 0
        self.Sgen = 0

    @property
    def Pr(self):
        # Prandtl number
        return self._bmat.nu * self._cmat.rho * self._cmat.c_p / self._cmat.k

    @property
    def N(self):
        # Number of channels
        return (self.W_d / 2 - self.w_w) // (self.w_c + self.w_w)

    @property
    def alpha(self):
        return 2 * self.w_c / self.H_c

    @property
    def beta(self):
        return self.w_c / self.w_w

    def warning(prop):
        return 'You cannot set ', prop, ' explicitly'

    def soundspeed(self):
        self.__csound = (self.gamma * self.Rgas * self.Ta) ** (1 / 2)
        return self.__csound

    def alphasubc(self):
        self.alpha = 2 * self.w_c / self.H_c
        return self.alpha

    def fbeta(self):
        self.beta = self.w_c / self.w_w
        return self.beta

    def numberchannels(self):
        self.N = round(((self.W_d / 2) - self.w_w) / (self.w_c + self.w_w))
        return self.N

    def hidraulicd(self):
        ac = 2 * self.w_c * self.H_c
        p = 2 * (self.H_c + 2 * self.w_c)
        self.__Dh = 4 * ac / p
        return self.__Dh

    def massflow(self):
        self.__mdot = self.rho * self.G / (2 * self.numberchannels())
        return self.__mdot

    def flowavv(self):
        self.Uavg = self.massflow() / (self.rho * self.w_c * self.H_c)
        return self.Uavg

    def machnum(self):
        self.__Ma = self.flowavv() / self.soundspeed()
        return self.__Ma

    def reynolds(self):
        self.ReDh = (self.hidraulicd() * self.flowavv()) / self.nu
        return self.ReDh

    def lsa(self):
        self.__Ab = self.W_d * self.L_d
        return self.__Ab

    def interfca(self):
        self.__Ai = self.W_i * self.L_i
        return self.__Ai

    def tht(self):
        self.__Q = self.q * self.interfca()
        return self.__Q

    def flowre(self):
        if (self.Regime != 'laminar') and (self.Regime != 'turbulent'):
            raise Exception('Regime must be laminar or turbulent')
        return self.Regime

    def frictionf(self):
        if (self.Regime == 'laminar'):
            B = ((self.alphasubc() ** 2) + 1) / ((self.alphasubc() + 1) ** 2)
            self.__f = ((3.2 * (self.reynolds() * self.hidraulicd() / self.L_d) ** (0.57)) ** 2 + (
                        4.7 + 19.64 * B) ** 2) ** (1 / 2) / self.reynolds()
        else:
            self.__f = 1 / (0.79 * np.log(self.reynolds()) - 1.64) ** 2
        return self.__f

    def nusseltnum(self):
        if (self.Regime == 'laminar'):
            self.__NuDh = 2.253 + 8.164 * (1 / (self.alphasubc() + 1)) ** (1.5)
        else:
            self.__NuDh = (self.frictionf() / 2) * (self.reynolds() - 1000) * self.Pr / (
                        1 + 12.7 * (self.frictionf() / 2) ** (0.5) * (self.Pr ** (2 / 3) - 1))
        return self.__NuDh

    def convection(self):
        self.__havg = self.kf * self.nusseltnum() / self.hidraulicd()
        return self.__havg

    def fineff(self):
        mhc = np.sqrt(2 * (2 * self.w_w + self.L_d) * self.convection() / (self.k * 2 * self.w_w * self.L_d)) * self.H_c
        self.__etap = np.tanh(mhc) / mhc
        return self.__etap

    def interfresis(self):
        self.__Ri = self.RicondA / self.interfca()
        return self.__Ri

    def convresis(self):
        self.__Rconv = 1 / (2 * (
                    self.numberchannels() + 1) * self.convection() * self.fineff() * self.H_c * self.L_d + 2 * self.numberchannels() * self.convection() * self.w_c * self.L_d)
        return self.__Rconv

    def fluidres(self):
        self.__Rf = 1 / (self.rho * self.G * self.cp)
        return self.__Rf

    def baseres(self):
        a = np.sqrt(self.interfca() / math.pi)
        b = np.sqrt(self.lsa() / math.pi)
        tau = self.H_b / b
        epsilon = a / b
        ro = self.interfresis() + self.convresis() + self.fluidres()
        bi = 1 / (math.pi * self.k * b * ro)
        lambdac = math.pi + 1 / (np.sqrt(math.pi) * epsilon)
        phic = (np.tanh(lambdac * tau) + lambdac / bi) / (1 + lambdac * np.tanh(lambdac * tau) / bi)
        psiavg = epsilon * tau / np.sqrt(math.pi) + 0.5 * phic * (1 - epsilon) ** (3 / 2)
        self.__Rb = psiavg / np.sqrt(math.pi) * self.k * a
        return self.__Rb

    def eqres(self):
        self.Req = self.interfresis() + self.baseres() + self.convresis() + self.fluidres()
        return self.Req

    def ftemp(self):
        self.__Tf = self.Ta + self.fluidres() * self.tht()
        return self.__Tf

    def suptemp(self):
        self.__Ts = self.ftemp() + self.convresis() * self.tht()
        return self.__Ts

    def basetemp(self):
        self.__Tb = self.suptemp() + self.baseres() * self.tht()
        return self.__Tb

    def inttemp(self):
        self.__Ti = self.basetemp() + self.interfresis() * self.tht()
        return self.__Ti

    def pressdrop(self):
        kce = 1.79 - 2.32 * (self.fbeta() / (1 + self.fbeta())) + 0.53 * (self.fbeta() / (1 + self.fbeta())) ** 2
        self.DeltaP = 0.5 * self.rho * self.flowavv() ** 2 * (self.frictionf() * (self.L_d / self.hidraulicd()) + kce)
        return self.DeltaP

    def flowpp(self):
        self.__Omega = self.G * self.pressdrop()
        return self.__Omega

    def egrdht(self):
        self.__Sgenht = (self.tht() ** 2) * self.eqres() / (self.Ta * self.inttemp())
        return self.__Sgenht

    def egrdff(self):
        self.__Sgenff = self.flowpp() / self.Ta
        return self.__Sgenff

    def tegr(self):
        self.Sgen = self.egrdht() + self.egrdff()
        return self.Sgen

    def Bejannum(self):
        self.__Be = self.egrdht() / self.tegr()
        return self.__Be

    def plotc(self):
        sgenp = np.arange(100)
        alphap = np.arange(100)
        betap = np.arange(100)

        sgenp = sgenp.astype(float)
        alphap = alphap.astype(float)
        betap = betap.astype(float)

        # wc = np.linspace(125e-6, 0.0004, 100)
        # sgenp = []
        wc = []

        # for wc_i in wc:
        #     self.wc = wc_i
        #     sgenp.append(self.tegr())

        for x in range(100):
            self.w_c = (250e-6 / 2) - (x * 1e-6)
            wc.append(self.w_c)
            sgenp[x] = self.tegr()
            alphap[x] = self.alpha
            betap[x] = self.beta

        ax = plt.axes(projection='3d')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        ax.set_zlabel(r'$\dot{S}_{gen}$')
        plt.plot(alphap, betap, sgenp)
        plt.show()

        # # ax = plt.axes(projection='3d')
        # plt.xlabel(r'$w_c$')
        # # plt.ylabel(r'$\beta$')
        # plt.ylabel(r'$\dot{S}_{gen}$')
        # plt.plot(np.array(wc), np.array(sgenp))
        # plt.show()


if __name__ == '__main__':
    mc = Microchannel()
    mc.plotc()
