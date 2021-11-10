#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:35:02 2021

@author: pablo
"""
import time, sys
import numpy as np
from sedpy.observate import load_filters
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from astropy.io import ascii

run_params = {'verbose': True,
              'debug': False,
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol': 0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # emcee fitting parameters
              # 'emcee':True,
              # 'nwalkers': 128,
              # 'nburn': [16, 32, 64],
              # 'niter': 500,
              # 'interval': 0.25,
              # 'initial_disp': 0.1,
              # dynesty Fitter parameters
              'dynesty': True,
              'nested_bound': 'multi',  # bounding method
              'nested_sample': 'unif',  # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.02,
              # 'nested_weight_kwargs': {"pfrac": 1.0},
              # 'nested_stop_kwargs': {"post_thresh": 0.1},
              # Obs data parameters
              'phottable':
                  'IllustrisTNG100_galaxy_bin_classification_subsample_fsps',
              'luminosity_distance': 10,  # in Mpc
              'snr': 100,
              'entry': 0,
              'object_redshift': 0.0,
              'sfh_bins': 7,  # NUMBER OF TIME BINS
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              # SPS parameters
              'zcontinuous': 1,
              }

# --------------
# Model Definition
# --------------


def build_model(object_redshift=None, fixed_metallicity=None, add_duste=False,
                add_neb=False, luminosity_distance=None, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models.templates import adjust_continuity_agebins
    from prospect.models import priors, sedmodel

    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at TemplateLibrary.describe("parametric_sfh") to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary['continuity_sfh']

    # Add lumdist parameter.  If this is not added then the distance is
    # controlled by the "zred" parameter and a WMAP9 cosmology.
    if luminosity_distance is not None:
        model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units": "Mpc"}

    model_params['logzsol']['isfree'] = False
    model_params['logzsol']['init'] = 0.0

    model_params["dust2"]["init"] = 0.0
    model_params["dust2"]["isfree"] = False

    model_params = adjust_continuity_agebins(model_params, tuniv=13.7,
                                             nbins=extras['sfh_bins'])
    if object_redshift is not None:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    # Now instantiate the model using this new dictionary
    model = sedmodel.SedModel(model_params)

    return model

# --------------
# Observational Data
# --------------


# Here we are going to put together some filter names
sdss = ['sdss_{0}0'.format(b) for b in 'ugriz']
filterset = (sdss)


def build_obs(entry=None, snr=None, phottable=None, luminosity_distance=None,
              **kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """

    from prospect.utils.obsutils import	fix_obs

    catalog = ascii.read(phottable)

    filternames = filterset
    M_AB = np.array([catalog['u_mag'][entry], catalog['g_mag'][entry],
                     catalog['r_mag'][entry], catalog['i_mag'][entry],
                     catalog['z_mag'][entry]])
    dm = 25 + 5.0 * np.log10(luminosity_distance)
    mags = M_AB + dm
    masks = np.ones_like(mags, dtype=bool)

    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects. See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies (Flux[JY]/3631 Jy).  It should have the same
    # order as `filters` above.
    obs["maggies"] = 10**(-0.4*mags)
    obs['maggies_unc'] = (1./snr) * obs["maggies"]
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = masks
    # We have no spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['entry'] = entry
    obs['SubhaloID'] = catalog['SubhaloID'][entry]

    # This ensures all required keys are present and adds some
    # extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags, dust1=0.)
    return sps

# -----------------
# Noise Model
# ------------------


def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------


def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__ == '__main__':

    run_params = run_params
    for i in range(24):
        run_params['entry'] = i
        # obs = build_obs(**run_params)
        # sps = build_sps(**run_params)
        # model = build_model(**run_params)
        # noise = build_noise(**run_params)
        obs, model, sps, noise = build_all(**run_params)

        run_params["sps_libraries"] = sps.ssp.libraries
        run_params["param_file"] = __file__

        # hfile = setup_h5(model=model, obs=obs, **run_params)
        hfile = "prospector_fits/illustris_subhaloID_{}_continuity_sfh.h5".format(
                obs['SubhaloID'])
        output = fit_model(obs, model, sps, noise, **run_params)

        writer.write_hdf5(hfile, run_params, model, obs,
                          output["sampling"][0], output["optimization"][0],
                          tsample=output["sampling"][1],
                          toptimize=output["optimization"][1],
                          sps=sps)
        try:
            hfile.close()
        except(AttributeError):
            pass
