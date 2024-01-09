#! /usr/bin/python
import numpy as np
from scipy.integrate import quad


def get_IFT(f, x, fargs):
    IFT_real_integrand = lambda k: np.real(f(k, *fargs)*np.exp(-1j*k*x)/(2*np.pi))
    IFT_imag_integrand = lambda k: np.imag(f(k,  *fargs)*np.exp(-1j*k*x)/(2*np.pi))
    IFT_real = quad(IFT_real_integrand, -np.inf, np.inf)[0]
    IFT_imag = quad(IFT_imag_integrand, -np.inf, np.inf)[0]
    return IFT_real + 1j*IFT_imag

def get_smearing_function(smearing, FT=False):
    if smearing.lower()=='gaussian':
        return (FT_gaussian_smearing if FT else gaussian_smearing)
    elif smearing.lower()=='lorentzian':
        return (FT_lorentzian_smearing if FT else lorentzian_smearing)
    elif smearing.lower()=='quartic':
        return (FT_quartic_smearing if FT else quartic_smearing)
    elif smearing.lower()=='sharp':
        return (FT_sharp_smearing if FT else sharp_smearing)
    elif smearing.lower()=='pointlike':
        return (FT_sharp_smearing if FT else pointlike_smearing)

def gaussian_smearing(x, sigma):
    return np.exp(-(x)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def lorentzian_smearing(x, sigma):
    return (sigma/(2*np.pi)) * 1/(x**2 + (sigma/2)**2)

def quartic_smearing(x, sigma):
    return (np.sqrt(2)*(sigma/2)**3/(np.pi)) * 1/(x**4 + (sigma/2)**4) 

def sharp_smearing(x, sigma):
    if abs(x) <= sigma/2:
        return 1/sigma
    else:
        return 0

def pointlike_smearing(x, sigma):
    if x==0:
        return 1/sigma
    else:
        return 0

def FT_gaussian_smearing(k, sigma):
    return np.exp(-(sigma*k)**2/4)

def FT_lorentzian_smearing(k, sigma):
    return np.exp(-sigma*abs(k)/2)/np.pi

def FT_quartic_smearing(k, sigma):
    a = sigma/np.sqrt(8)
    return (1/np.pi)*np.exp(-np.abs(k)*a)*(np.sin(a*abs(k)) + np.cos(a*abs(k)))

def FT_sharp_smearing(k, sigma):
    return np.sin(k*sigma/2)/(k*sigma/2)

def get_cutoff_function_FT(cutoff):
    if cutoff.lower()=='gaussian':
        return FT_gaussian_cutoff
    elif cutoff.lower()=='lorentzian':
        return FT_lorentzian_cutoff
    elif cutoff.lower()=='exponential':
        return FT_exponential_cutoff
    elif cutoff.lower()=='sharp':
        return FT_sharp_cutoff

def FT_gaussian_cutoff(k, eps, omega):
    if np.abs(k) <= omega:
        return 1
    else:
        return np.exp(-(((np.abs(k) - omega)/eps)**2)/2)

def FT_lorentzian_cutoff(k, eps, omega):
    if np.abs(k) <= omega:
        return 1
    else:
        return eps**2/(eps**2 + (np.abs(k) - omega)**2)

def FT_exponential_cutoff(k, eps, omega):
    if np.abs(k) <= omega:
        return 1
    else:
        return np.exp(-(np.abs(k) - omega)/(np.sqrt(2)*eps))

def FT_sharp_cutoff(k, eps, omega):
    if np.abs(k) <= omega:
        return 1
    else:
        return 0

