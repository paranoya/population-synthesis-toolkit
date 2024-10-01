"""
Dust Extinction/Emission Models for Stellar Populations

This module implements dust extinction and emission models for stellar population synthesis,
including a base class for general dust models and specific implementations such as a dust screen 
and the Charlot & Fall (2000) extinction model.

Usage
-----
This module is intended for applying dust extinction or emission to stellar spectra, either
to synthetic stellar populations or other types of spectra.
"""
from astropy import units as u
import numpy as np
import extinction

from abc import ABC, abstractmethod


class DustModelBase(ABC):
    """
    Abstract base class for dust extinction and emission models.

    Description
    -----------
    This class provides the framework for implementing dust models. Subclasses
    should define methods to compute the extinction and emission due to dust.
    
    Attributes
    ----------
    extinction_law : str
        The name of the extinction law to be used. This is retrieved from the 
        `extinction` library.
    """

    @abstractmethod
    def get_extinction(self, *args, **kwargs):
        """
        Compute the dust extinction for a given set of parameters.
        
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_emission(self, *args, **kwargs):
        """
        Compute the dust emission for a given set of parameters.
        
        This method must be implemented by subclasses.
        """
        pass

    def apply_extinction(self, wavelength, spectra, axis=-1, **kwargs):
        """
        Apply the dust extinction model to a given spectra.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array. Can be either a numpy array of floats or an `astropy.Quantity`
            with associated units (e.g., Angstroms).
        spectra : np.ndarray or astropy.Quantity
            Array of spectra to which the extinction will be applied.
        axis : int, optional
            The axis of the spectra array corresponding to the wavelength dimension. Default is -1.
        **kwargs
            Additional keyword arguments passed to the `get_extinction` method.

        Returns
        -------
        reddened_spectra : np.ndarray
            The spectra array with dust extinction applied.
        """
        ext = self.get_extinction(wavelength, **kwargs)
        if ext.ndim != spectra.ndim:
            new_dims = tuple(np.delete(np.arange(spectra.ndim), axis))
            ext = np.expand_dims(ext, new_dims)
        return spectra * ext

    def apply_emission(self, wavelength, spectra, axis=-1, **kwargs):
        """
        Add the predicted dust emission to a given spectra.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array. Can be either a numpy array of floats or an `astropy.Quantity`
            with associated units.
        spectra : np.ndarray or astropy.Quantity
            Array of spectra to which the dust emission will be added.
        axis : int, optional
            The axis of the spectra array corresponding to the wavelength dimension. Default is -1.
        **kwargs
            Additional keyword arguments passed to the `get_emission` method.

        Returns
        -------
        spectra_with_emission : np.ndarray
            The spectra array with dust emission added.
        """
        emission = self.get_emission(wavelength, **kwargs)
        if emission.ndim != spectra.ndim:
            new_dims = tuple(np.delete(np.arange(spectra.ndim), axis))
            emission = np.expand_dims(emission, new_dims)
        return spectra + emission
        
    def redden_ssp_model(self, ssp_model, **kwargs):
        """
        Apply extinction to a simple stellar population (SSP) model.

        Parameters
        ----------
        ssp_model : `pst.SSPBase` object
            A simple stellar population (SSP) model instance.
        **kwargs
            Additional keyword arguments passed to the `apply_extinction` method.

        Returns
        -------
        reddened_ssp_model : `pst.SSPBase` object
            The SSP model with dust extinction applied.
        """
        reddened_ssp_model = ssp_model.copy()
        reddened_ssp_model.L_lambda = self.apply_extinction(
            ssp_model.wavelength, reddened_ssp_model.L_lambda, axis=-1, **kwargs)
        return reddened_ssp_model


class DustScreen(DustModelBase):
    """
    Dust screen extinction model.

    Implements a simple dust screen model where dust extinction is applied
    to spectra using a specified extinction law and R_V parameter.

    Attributes
    ----------
    extinction_law_name : str
        The name of the extinction law from the `extinction` library (e.g., 'ccm89', 'odonnell94').
    r_extinction : float
        The R_V value for the extinction law. Default is 3.1.
    """
    def __init__(self, extinction_law_name, r_extinction=3.1):
        # super().__init__(extinction_law)
        self.extinction_law_name = extinction_law_name
        self.r_extinction = r_extinction

        self.extinction_law = getattr(extinction, self.extinction_law_name)

    def get_extinction(self, wavelength, a_v=1.0):
        """
        Compute the dust extinction.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array in Angstroms.
        a_v : float, optional
            The V-band extinction (in magnitudes). Default is 1.0.

        Returns
        -------
        extinction_curve : np.ndarray
            Dimensionless extinction factor to be applied to the spectra.
        """
        return 10**(-0.4 * self.extinction_law(
            np.array(wavelength.to_value("angstrom"), dtype=float),
            a_v, self.r_extinction)) <<  u.dimensionless_unscaled

    def get_emission(self, wavelength):
        """
        Compute the dust emission.

        For this model, no dust emission is included, so this method returns zeros.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array.

        Returns
        -------
        emission : np.ndarray
            An array of zeros with the same shape as `wavelength`.
        """
        return np.zeros(wavelength.size)

class CF03DustScreen(DustScreen):
    """
    Charlot & Fall (2000) dust screen model for young and old stellar populations.

    This model applies different extinction curves to young and old populations
    based on their ages.

    Parameters
    ----------
    extinction_law_name : str
        The name of the extinction law from the `extinction` library.
    young_ssp_age : astropy.Quantity
        The age threshold for defining young populations (in years).
    r_extinction : float, optional
        The R_V value for the extinction law. Default is 3.1.
    """
    def __init__(self, extinction_law_name, young_ssp_age, r_extinction=3.1):
        assert isinstance(young_ssp_age, u.Quantity), "young_ssp_age must be an astropy.Quantity"
        self.young_ssp_age = young_ssp_age
        super().__init__(extinction_law_name, r_extinction=r_extinction)
    
    def get_extinction(self, wavelength, age, a_v_young=1.0, a_v_old=0.3):
        """
        Compute the dust extinction for young and old stellar populations.

        Parameters
        ----------
        wavelength : np.ndarray or astropy.Quantity
            Wavelength array.
        age : np.ndarray or astropy.Quantity
            Array of stellar population ages.
        a_v_young : float, optional
            V-band extinction for young populations. Default is 1.0.
        a_v_old : float, optional
            V-band extinction for old populations. Default is 0.3.

        Returns
        -------
        extinction_curve : np.ndarray
            2D array of extinction factors with shape (age.size, wavelength.size).
        """
        age = np.atleast_1d(age)
        young = age < self.young_ssp_age
        ext = np.zeros((age.size, wavelength.size))
        ext[young] = super().get_extinction(wavelength, a_v_young) 
        ext[~young] = super().get_extinction(wavelength, a_v_old)
        return ext

        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    # Redden some set of spectra using Charlote and Fall 03 model
    dust_model = CF03DustScreen("ccm89", young_ssp_age=10 * u.yr)
    
    wavelength = np.linspace(1000, 10000) * u.angstrom
    spectra = np.ones((10, wavelength.size))
    ages = np.linspace(5, 15, 10) * u.yr
    reddened_spectra = dust_model.apply_extinction(wavelength, spectra,
                                                   age=ages,
                                                   a_v_young=1.0, a_v_old=0.3)
    
    plt.figure()
    plt.title("Charlote and Fall 03 dust extinction model")
    plt.plot(wavelength, spectra[0], label=f'Unreddened')
    plt.plot(wavelength, reddened_spectra[0], label=f'Age={ages[0]}')
    plt.plot(wavelength, reddened_spectra[-1], label=f'Age={ages[-1]}')
    plt.legend()
    
    # Apply the extinction to a given SSP model
    from pst.SSP import PopStar
    ssp = PopStar(IMF='cha')
    dust_model = DustScreen("ccm89",)
    red_ssp = dust_model.redden_ssp_model(ssp, a_v=1.0)
    
    plt.figure()
    plt.title("Redden SSP model")
    plt.loglog(ssp.wavelength, ssp.L_lambda[3, -1])
    plt.loglog(ssp.wavelength, red_ssp.L_lambda[3, -1])
    plt.xlim(800, 1e5)
    plt.ylim(1e-8, 1e-4)
    
    # Little performance test
    from time import time
    a_v = np.linspace(0.1, 3, 1)
    tstart = time()
    ssps = [dust_model.redden_ssp_model(ssp, a_v=av) for av in a_v]
    tend = time()
    print(f"Time for generating {a_v.size} SSP models: {tend - tstart}")
