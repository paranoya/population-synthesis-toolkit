import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from astropy import units as u

from pst import models
from pst import SSP

today = 13.7 * u.Gyr
cosmic_time = np.geomspace(0.1 * u.Gyr, today)
lookback_time = today - cosmic_time

mass_history = (1 - np.exp(-cosmic_time / 3 / u.Gyr)) * u.Msun

# Monotonically increasing metallicity

z_histories = [0.02 * mass_history / mass_history[-1],
             0.02 * (1 - mass_history / mass_history[-1]),
            (0.02 * np.sin(cosmic_time.to_value("Gyr") / 0.5)**2 + 0.001
              ) * u.dimensionless_unscaled
]
z_labels = ['Monotonically increasing Z(t)', 'Monotonically decreasing Z(t)',
            'Oscillating Z(t)']

# Monotonically decreasing metallicity

ssp = SSP.PopStar(IMF='cha')

fig, axs = plt.subplots(nrows=3, ncols=1, constrained_layout=True,
                        sharex=True, sharey=True)
color_mesh_args = {"vmin": -5, "vmax": 1, "cmap": "nipy_spectral"}

for z_hist, label, z_axs in zip(z_histories, z_labels, axs):

    sfh_model = models.Tabular_CEM(cosmic_time, mass_history, z_hist)

    #old_masses = models.ChemicalEvolutionModel.interpolate_ssp_masses(sfh_model, ssp, today)
    new_masses = sfh_model.interpolate_ssp_masses(ssp, t_obs=today)

    plt.figure(fig)
    '''
    ax = z_axs[0]
    ax.set_title("Old method " + label)
    mappable = ax.pcolormesh(ssp.ages.to_value("Gyr"),
                    np.log10(ssp.metallicities.value),
                    np.log10(old_masses.value),
                    **color_mesh_args)
    ax.plot(lookback_time, np.log10(z_hist), '-o', color='fuchsia', label='Input Z(t)')
    ax.legend()
    ax.set_xlabel("Lookback time (Gyr)")
    ax.set_ylabel("Z(t)")
    plt.colorbar(mappable, ax=ax, label=r'$\log_{10}(M/Msun)$')
    ax = z_axs[1]
    '''
    ax = z_axs
    ax.set_title("New method " + label)
    mappable = ax.pcolormesh(ssp.ages.to_value("Gyr"),
                    np.log10(ssp.metallicities.value),
                    np.log10(new_masses.value),
                    **color_mesh_args)
    ax.plot(lookback_time, np.log10(z_hist), '-o', color='fuchsia')
    plt.colorbar(mappable, ax=ax, label=r'$\log_{10}(M/Msun)$')
    
    for a0 in ssp.ages:
        ax.axvline(a0.to_value(u.Gyr), color='c', ls='-', alpha=.1)
        
    for z0 in ssp.metallicities:
        ax.axhline(np.log10(z0), color='c', ls='-', alpha=.1)
        

    a = np.sort(np.hstack([
        np.geomspace(1*u.yr, today, 1000),
        np.linspace(1*u.yr, today, 1000),
    ]))
    ax.plot(a,  np.log10(sfh_model.ism_metallicity(today-a)), 'w--')
    
    ax.set_xlabel("Lookback time (Gyr)")
    ax.set_xlim(ssp.ages[0].to_value("Gyr")/2, ssp.ages[-1].to_value("Gyr")*2)
    ax.set_xscale('log')
    ax.set_ylabel(r"$\log_{10} Z(t)$")
    ax.set_ylim(np.log10(ssp.metallicities[0])-.3, np.log10(ssp.metallicities[-1])+.3)
plt.show()
