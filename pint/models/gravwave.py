"""This module implements gravitational waves.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import astropy.units as u
from .timing_model import GravitationalWaveComponent
from . import parameter
import math, os
import scipy.interpolate as interp
import astropy.constants as c 
import pint as p 


class GravWave(GravitationalWaveComponent):
    """This is a class to implement gravitational waves
    """
    register = True
    def __init__(self):
        super(GravWave, self).__init__()
        self.add_param(parameter.floatParameter(name = 'CMASS', units='solMass', 
            description="Chirp mass"))
        
        self.grav_funcs_component += [self.grav_wave,]
        self.category = 'grav_wave'

    # def grav_wave(self):

    #     """This is a test function to see how this implements into PINT.
    #     """
    #     print(self.CMASS.quantity)
    #     CMASS = pint.Tsun / self.CMASS.quantity
    #     print(CMASS)
    #     print("Hey")

    def grav_wave(pint_toa, pint_model, gwtheta, gwphi, mc, dist, fgw, phase0, psi,
                inc, pdist=1.0, pphase=None, psrTerm=True,
                evolve=True, phase_approx=False, tref=0):
        ''' 
        This function accepts a series of parameters for a continuous
        gravitational wave and returns the timing residuals caused by 
        said wave.
        The method of calculating the residuals is defined by Ellis et. al
        2012; this code is modeled after the libstempo package written by 
        Stephen Taylor (? and other people?)
        
        Will need to read in toas and .par file
        Will have to change a lot of stuff to -> self.stuff (self.RAJ.quantity)
        Parameters
        ----------
        psr : libstempo class # want to change to pint model class
            Pulsar object
        gwtheta : float # int?
            Polar angle of GW source in celestial coordinates [units: radians]
        gwphi : float # int?
            Azimuthal angle of GW source in celestial coordinates [units: radians]
        mc : float # ? do np.double for all floats
            Chirp mass of SMBHB [solar masses]
        dist : float # ?
            Luminosity distance to SMBHB [Mpc]
        fgw : float # ?
            GW frequency, twice the orbital frequency [Hz]
        phase0 : float # ?
            Iniitial phase of GW source [radians]
        psi: float # ?
            Polarization of GW source [radians]
        inc : double
            Inclination of GW source [radians]
        pdist : float # ?
            Pulsar distance to use other than those in psr [kpc]
        pphase : float # ?
            Use pulsar phase to determine distance [radians]
        psrTerm : bool
            Option to include pulsar term [bool]
        evolve : bool
            Option to exclude evolution [bool]
        tref : ?
            Fiducial time at which initial parameters are referenced [?]
        Returns
        -------
        Vector of induced residuals
        '''
        # convert units 
        mc *= p.Tsun # chirp mass in solar mass -> seconds
        dist *= (1*u.Mpc).to(p.ls) # distance to SMBHB in Mpc -> seconds
        # defining initial orbital frequency
        w0 = np.pi * fgw
        phase0 /= 2 # orbital phase
        w053 = w0**(-5/3)
        # define variable for later use 
        # cosgwtheta = np.cos(gwtheta)
        # cosgwphi = np.cos(gwphi)
        # singwtheta = np.sin(gwtheta)
        # singwphi = np.sin(gwphi)
        # sin2psi = np.sin(2*psi)
        # cos2psi = np.cos(2*psi)
        # incfac1 = 0.5*(3+np.cos(2*inc))
        # incfac2 = 2*np.cos(inc)

        cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
        singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
        sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)
        incfac1, incfac2 = 0.5*(3+np.cos(2*inc)), 2*np.cos(inc)
        
        # unit vectors to GW source 
        m = np.array([singwphi, -cosgwphi, 0.0])
        n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
        omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])
        # various factors involving GW parameters
        fac1 = 256/5 * mc**(5/3) * w0**(8/3)
        fac2 = 1/32/mc**(5/3)
        fac3 = mc**(5/3)/dist
        # in PINT m.DECJ.units is deg
        # then m.DECJ.quantity.to(u.rad) puts it into rads 
        # then change m to self - assuming this is going into a class
        # pulsar location ## TODO this might not hold in PINT, ra and dec stuff? ---- double check the name of par file
        if 'RAJ' and 'DECJ' in pint_model.params: 
            ptheta = np.pi/2 - pint_model.DECJ.value 
            pphi = pint_model.RAJ.value
        elif 'ELONG' and 'ELAT' in pint_model.params:
            icrs_coords = pint_model.coords_as_ICRS()
            ptheta = np.pi/2 - (icrs_coords.dec.to(u.rad)).value
            pphi = (icrs_coords.ra.to(u.rad)).value
        # use definitions from Sesana et. al 2010 and Ellis et. al 2012
        phat = np.array([np.sin(ptheta)*np.cos(pphi),np.sin(ptheta)*np.sin(pphi),
                np.cos(ptheta)])
        fplus = 0.5 * (np.dot(m, phat))**2 - np.dot(n, phat)**2 / (1+np.dot(omhat,phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1+np.dot(omhat, phat))
        cosMu = -np.dot(omhat, phat)
        # defining and calling in the toas
        gettoas = pint_toa.get_mjds()
        gettoas = np.asarray(gettoas)
        # get values from pulsar object 
        toas = np.double(gettoas)*86400 - tref 
        if pphase is not None:
            pd = pphase/(2*np.pi*fgw*(1-cosMu)) / ((1*u.kpc).to(u.s,equivalencies=p.light_second_equivalency)).value
        else:
            pd = pdist
        # convert units 
        pd *= ((1*u.kpc).to(u.s,equivalencies=p.light_second_equivalency)).value
        # get pulsar time
        tp = toas-pd*(1-cosMu)
        # evolution
        if evolve:
            # calculate time dependent frequency at earth and pulsar
            omega = w0 * (1 - fac1.value * toas)**(-3/8)
            omega_p = w0 * (1 - fac1.value * tp)**(-3/8)
            # calculate time dependent phase
            phase = phase0 + fac2.value * (w053 - omega**(-5/3))
            phase_p = phase0 + fac2.value * (w053 - omega_p**(-5/3))
        # use approximation that frequency does not evolve over obs. time
        elif phase_approx:
            # frequencies
            omega = w0
            omega_p = w0 * (1 + fac1 * pd*(1-cosMu))**(-3/8)
            
            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3)) + omega_p*toas
              
        # no evolution
        else: 
            
            # monochromatic
            omega = w0
            omega_p = omega
            
            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + omega * tp
            
        # define time dependent coefficients
        At = np.sin(2*phase) * incfac1
        Bt = np.cos(2*phase) * incfac2
        At_p = np.sin(2*phase_p) * incfac1
        Bt_p = np.cos(2*phase_p) * incfac2
        # now define time dependent amplitudes 
        alpha = fac3.value / omega**(1/3)
        alpha_p = fac3.value / omega_p**(1/3)

        # define rplus and rcross
        rplus = alpha * (At*cos2psi + Bt*sin2psi)
        rcross = alpha * (-At*sin2psi + Bt*cos2psi)
        rplus_p = alpha_p * (At_p*cos2psi + Bt_p*sin2psi)
        rcross_p = alpha_p * (-At_p*sin2psi + Bt_p*cos2psi)
        # residuals
        if psrTerm:
            res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
        else:
            res = -fplus*rplus - fcross*rcross # don't need to add this to add to residuals already
        return res 
