"""Make plots to view the outcome of a sensitivty test."""
# %% 
# Note: don't try and generate the sensitivity result on a local machine, it will take too long!
# Make sure it is plotting from a file
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import xarray as xr
import numpy as np

from simtrails.misc.plotting import contour_taus
from simtrails.sensitivity_result import SensitivityResult

RESOLUTION = 2

sensitivity = SensitivityResult.from_name("optical_thickness_width")
sensitivity.add_eff_width()

fig, axs = plt.subplots(5,6, figsize=(20, 15), constrained_layout=True)
for ax in axs.flat:
    ax.set_xscale("log")
    ax.set_yscale("log")

sensitivity.plot_2d("iwp", "eff_radius", hue_var="eff_width", ax=axs.flat, resolution=RESOLUTION, length=150, norm=Normalize(0, 3.))

min_width = (
        xr.where(
            (sensitivity.data.sel(resolution=RESOLUTION).eff_width > (RESOLUTION / 2)).mean("repeat") > 0.5,
            sensitivity.data.width,
            np.inf,
        )
        .min("width")
        .squeeze()
    )
fig, ax = plt.subplots()
min_width.plot(ax=ax, norm=Normalize(0, 3.), cmap="cividis_r")
ax.set_xscale("log")
ax.set_yscale("log")
contour_taus(ax, "iwp", "eff_radius")
ax.set_xlabel("IWP$_0$ [g m$^{-2}$]")
ax.set_ylabel("Effective radius [um]")
ax.set_title(f"Minimum width for detection of a Gaussian contrail\nusing a {RESOLUTION} km imager")
fig.show()

# %%
