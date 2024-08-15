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

fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True,
                        sharex=True, sharey=True)
color_mesh_args = {"vmin": -5, "vmax": 1, "cmap": "nipy_spectral"}

for z_hist, label, z_axs in zip(z_histories, z_labels, axs):

    sfh_model = models.Tabular_MFH(times=cosmic_time, masses=mass_history,
                                   Z=z_hist)

    old_masses = sfh_model.interpolate_ssp_masses(ssp, t_obs=today)
    new_masses = sfh_model.interpolate_ssp_masses_new(ssp, t_obs=today)

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
    ax.set_title("New method " + label)
    mappable = ax.pcolormesh(ssp.ages.to_value("Gyr"),
                    np.log10(ssp.metallicities.value),
                    np.log10(new_masses.value),
                    **color_mesh_args)
    ax.plot(lookback_time, np.log10(z_hist), '-o', color='fuchsia')
    plt.colorbar(mappable, ax=ax, label=r'$\log_{10}(M/Msun)$')

    ax.set_xlabel("Lookback time (Gyr)")
    ax.set_ylabel("Z(t)")
plt.show()