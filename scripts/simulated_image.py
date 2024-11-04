"""Simulate an image of a Gaussian contrail using GOES-R ABI."""
# %%
import matplotlib.pyplot as plt
import numpy as np

from simtrails.contrail import GaussianContrail
from simtrails.imager import Imager

goes_imager = Imager.from_name("GOES_R_ABI")

contrail = GaussianContrail()

obs = goes_imager.simulate_observation(contrail, 14)

x_vals = np.arange(0, obs.shape[1]) * goes_imager.channels[14].resolution
y_vals = np.arange(0, obs.shape[0]) * goes_imager.channels[14].resolution

fig, ax = plt.subplots(figsize=(6, 4))

h = ax.pcolormesh(x_vals, y_vals, obs, cmap="Blues")
plt.colorbar(h, ax=ax, label="Brightness Temperature [K]")

ax.set_aspect("equal")
ax.set_title("Simulated GOES-R ABI ch. 14 observation")

ax.set_xlabel("Distance [km]")
ax.set_ylabel("Distance [km]")
fig.show()
# %%
