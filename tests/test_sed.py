import unittest
import numpy as np
from astropy import units as u
from astropy import constants
from pst import dust, SSP, sed, models, galaxy
from pst.observables import load_photometric_filters

class TestGalaxySED(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        print("Setting SSP model for testing dust models")

        ssp = SSP.PopStar(IMF="cha")
        # ssp = SSP.BC03_2016()
        sfh = models.ExponentialDelayedCEM(tau=10, today=13.7, mass_today=1e10,
        ism_metallicity_today=0.02)
        self.stellar_em = sed.StellarComponent(ssp=ssp, sfh=sfh)
        self.dust_att = dust.DustScreenAttenuation(a_v=1.0 << u.mag)
        
        self.dust_em = dust.CalorimetricDustComponent(
            attenuation=self.dust_att,
            dust_sed_component=dust.Casey2012DustComponent())

    def test_model(self):

        filters = load_photometric_filters(["Euclid_VIS.vis", "Euclid_NISP.Y",
                                            "WISE_WISE.W1", 
                                            "WISE_WISE.W2",
                                            "WISE_WISE.W3",
                                            "WISE_WISE.W4"],
                                            to_filter_list=True)

        model = galaxy.GalaxySED(stellar_model=self.stellar_em,
                                dust_attenuation_model=self.dust_att,
                                dust_model=self.dust_em,
                                #   target_wavelength=self.stellar_em.ssp.wavelength,
                                redshift=0.5, cosmology=None,
                                filters=filters)

        params = model.build_param_index(include_fixed=True)
        print(params)

        new_params = {"redshift": 0.2}
        model.update_parameters(new_params)
        update = model.parameters_recursive()
        print(update)
        spec = model.emission_components(dust_att_params=dict())

        spec_tot = model.emission_spectrum(dust_att_params=dict(a_v=1.0),
                                            to_obs_frame=False)
        
        self.assertTrue(np.isfinite(spec_tot).all())
        ctot = spec["stellar_sed"] + spec["dust_sed"]
        tot_rf = (spec_tot).to(ctot.unit)
        self.assertTrue(np.allclose(ctot, tot_rf))

        # Compute photometry
        phot_tot = model.emission_photometry(dust_att_params=dict(a_v=1.0),
                                             to_obs_frame=True)
        mags = -2.5 * np.log10(phot_tot.to_value("3631 Jy"))

        self.assertTrue(np.isfinite(mags).all()
                        & (mags > 12).all() & (mags < 30).all())

        spec_tot = model.emission_spectrum(dust_att_params=dict(a_v=1.0),
                                            to_obs_frame=True)
        self.assertTrue(np.isfinite(spec_tot).all())

if __name__ == '__main__':
    unittest.main()