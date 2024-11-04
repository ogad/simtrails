"""Run the Mannstein detector over images with different resolutions."""
# %%
import matplotlib.pyplot as plt

from simtrails.contrail import GaussianContrail
from simtrails.contrail_detector import GOESMannsteinDetector


resolutions = [0.5, 2., 7.]

contrail = GaussianContrail(length=150)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for resolution, ax in zip(resolutions, axs):
    detector = GOESMannsteinDetector(resolution=resolution)
    detector.contrail_mask(contrail, ax=ax)
    
    ax.set_title(f"Resolution: {resolution} km")
    ax.set_aspect("equal")
    
fig.show()

# %%
