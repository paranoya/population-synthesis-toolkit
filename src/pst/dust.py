from astropy import units as u
import numpy as np
import extinction

from abc import ABC, abstractclassmethod


class DustModelBase(ABC):
    """Dust extinction/emission base model.
    
    Description
    -----------
    
    Attributes
    ----------
    - extinction_law: str
        Name of the extinction law to be used. This is retrieved from the 
        `extinction` library.
    """

    @abstractclassmethod
    def get_extinction(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def get_emission(self, *args, **kwargs):
        pass

    def apply_extinction(self, wavelength, spectra, axis=-1, **kwargs):
        """Apply the dust extinction model to a given spectra
        
        Description
        -----------
        
        TODO: Write equation...
        
        Parameters
        ----------
        - wavelength: np.ndarray or astropy.Quantity
            Wavelength array.
        - spectra: np.ndarray or astropy.Quantity
            ...
        - axis: int
            ...
        - **kwargs:
            Key-word arguments to be passed to `get_extinction`.
        
        Returns
        -------
        # TODO: This could even include extinction but also adding the 
        re-emission
        - reddened_spectra: np.ndarray
            Reddened spectra.
        """
        ext = self.get_extinction(wavelength, **kwargs)
        if ext.ndim != spectra.ndim:
            new_dims = tuple(np.delete(np.arange(spectra.ndim), axis))
            ext = np.expand_dims(ext, new_dims)
        return spectra * ext

    def apply_emission(self, wavelength, spectra, axis=-1, **kwargs):
        """Add the predicted dust emission to a given spectra
        
        Description
        -----------
        
        TODO:
        
        Parameters
        ----------
        - wavelength: np.ndarray or astropy.Quantity
            Wavelength array.
        - spectra: np.ndarray or astropy.Quantity
            ...
        - axis: int
            ...
        - **kwargs:
            Key-word arguments to be passed to `get_extinction`.
        
        Returns
        -------
        # TODO: This could even include extinction but also adding the 
        re-emission
        - reddened_spectra: np.ndarray
            Reddened spectra.
        """
        emission = self.get_emission(wavelength, **kwargs)
        if emission.ndim != spectra.ndim:
            new_dims = tuple(np.delete(np.arange(spectra.ndim), axis))
            emission = np.expand_dims(emission, new_dims)
        return spectra + emission
        
    def redden_ssp_model(self, ssp_model, **kwargs):
        reddened_ssp_model = ssp_model.copy()
        reddened_ssp_model.L_lambda = self.apply_extinction(
            ssp_model.wavelength, reddened_ssp_model.L_lambda, axis=-1, **kwargs)
        return reddened_ssp_model


class DustCalorimetricModel():
    pass


class DustScreen(DustModelBase):
    """Dust screen extinction model.
    
    Attributes
    ----------
    - extinction_law: str
        N
    """
    def __init__(self, extinction_law_name, r_extinction=3.1):
        # super().__init__(extinction_law)
        self.extinction_law_name = extinction_law_name
        self.r_extinction = r_extinction

        self.extinction_law = getattr(extinction, self.extinction_law_name)

    def get_extinction(self, wavelength, a_v=1.0):
        return 10**(-0.4 * self.extinction_law(
            wavelength.to_value("angstrom"), a_v, self.r_extinction)
            ) <<  u.dimensionless_unscaled

    def get_emission(self, wavelength):
        return np.zeros(wavelength.size)

class CF03DustScreen(DustScreen):
    def __init__(self, extinction_law_name, young_ssp_age, r_extinction=3.1):
        assert isinstance(young_ssp_age, u.Quantity), "young_ssp_age must be an astropy.Quantity"
        self.young_ssp_age = young_ssp_age
        super().__init__(extinction_law_name, r_extinction=r_extinction)
    
    def get_extinction(self, wavelength, age, a_v_young=1.0, a_v_old=0.3):
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
