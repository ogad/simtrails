"""Run a simple sensitivity test varying IWP and resolution."""
# %%
import numpy as np
import platform, multiprocessing
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from simtrails.contrail import GaussianContrail
from simtrails.contrail_detector import GOESMannsteinDetector
from simtrails.misc.plotting import contour_taus
from simtrails.sensitivity_test import CDASensitivityTest

if  __name__ == "__main__":
    if platform.system() == "Darwin":
        try:
            multiprocessing.set_start_method('spawn')
        except:
            pass
    test = CDASensitivityTest(GOESMannsteinDetector, GaussianContrail)
    result = test.repeats(4, {"iwp": np.logspace(-1, 1.5, 10), "resolution":[0.5, 2., 7.], "width":[2]})
    
    result.add_eff_width()
    
    fig, ax = plt.subplots()
    result.plot_series(ax=ax, var1="iwp", y_var="eff_width")
    ax.set_xscale("log")
    
    ax.set_xlabel("IWP$_0$ [g m$^{-2}$]")
    ax.set_ylabel("Detected effective width [km]")
    contour_taus(ax=ax, x_var="iwp", y_var=None, eff_radius=10)
    
    handles, labels = ax.get_legend_handles_labels()
    
    # add dotted line to the legend
    handles.append(Line2D([0], [0], color="black", linestyle="--"))
    labels.append("Optical thickness")
    for i in range(3):
        labels[i] = labels[i] + " km resolution"
    ax.legend(handles, labels)
    
    ax.set_title("Sensitivity of contrail detection to IWP$_0$")


# %%
