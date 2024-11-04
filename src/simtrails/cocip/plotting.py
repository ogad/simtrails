from matplotlib.colors import CenteredNorm, ListedColormap, SymLogNorm, Normalize
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import LogLocator
from simtrails.cocip.analysis import is_logspace
import numpy as np
from simtrails.misc.plotting import colorbar, contour_taus
import seaborn as sns

from .analysis import histogram

OBS_HUE_ORDER = ["too optically thin", "too narrow", "observable"]


def plot_contrails(df, lonlat=None):
    if lonlat is not None:
        df = df[df.longitude.between(*lonlat[0]) & df.latitude.between(*lonlat[1])]

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 5)
    )

    for flight_id in df.flight_id.unique():
        df[df.flight_id == flight_id].plot.scatter(
            "longitude",
            "latitude",
            # alpha="tau_contrail",
            c="rf_net",
            cmap="coolwarm",
            norm=plt.cm.colors.CenteredNorm(0, 30),
            ax=ax,
            colorbar=False,
            s=0.05,
        )
    ax.coastlines()
    colorbar(
        plt.cm.ScalarMappable(norm=plt.cm.colors.CenteredNorm(0, 30), cmap="coolwarm"),
        ax=ax,
    )

    if lonlat is not None:
        ax.set_extent([*lonlat[0], *lonlat[1]])
    return ax


def plot_2d_hist(
    ax,
    df,
    rf=False,
    use_eff=False,
    iwc_col="iwc",
    cb=True,
    lims=True,
    return_hist=False,
):
    import numpy as np

    iwc_labels = {
        "iwc": "IWC (\\unit{\\kilo\\gram\\per\\kilo\\gram})",
        "iwc_gm3": "IWC (\\unit{\\gram\\per\\metre\\cubed})",
        "iwp": "IWP (\\unit{\\gram\\per\\metre\\squared})",
    }
    iwc_upper_lims = {
        "iwc": 3e-5,
        "iwc_gm3": 0.015,
        "iwp": 15,
    }

    radius_col = "eff_radius" if use_eff else "r_ice_vol"

    hist, (xs, ys) = histogram(
        df,
        [iwc_col, radius_col],
        [10000, 100],
        rf=rf,
    )

    ys *= 1e6

    if rf:
        cmap = plt.get_cmap("coolwarm")
        cmap.set_over("darkred")
        cmap.set_under("darkblue")
        h = ax.pcolormesh(
            xs,
            ys,
            hist.T,
            cmap=cmap,
            norm=CenteredNorm(0),  # , 1000),
        )
    else:
        cmap = plt.get_cmap("Greens")
        cmap.set_over("darkgreen")
        h = ax.pcolormesh(xs, ys, hist.T, cmap=cmap, norm=Normalize(0))  # , 1000)

    ax.set_xlabel(iwc_labels[iwc_col])
    ax.set_ylabel(
        f"{'Volume' if not use_eff else 'Effective'} radius (\\unit{{\\micro\\metre}})"
    )

    if lims:
        ax.set_xlim(0, iwc_upper_lims[iwc_col])
        ax.set_ylim(0, 30)

    if cb:
        cax = colorbar(h, ax=ax)
        cax.set_ylabel(r"RF (W/m2) $\times$ count" if rf else "count")

    if return_hist:
        return hist, xs, ys
    return h


def plot_observability_histogram(
    histograms,
    category,
    hist_kwargs={},
    ax=None,
    obs_contour=True,
    tau_contour=True,
    cbar=True,
    label_cbar=True,
    peak=None,
    normalize=False,
):
    labels = {
        "observability": "Detection probability",
        "occurrence": "Contrail segments",
        "rf": "RF-weighted segments",
        "rf_sw": "SW RF-weighted segments",
        "rf_lw": "LW RF-weighted segments",
    }
    units = {
        "observability": "",  # "\\unit{{px}}",
        "occurrence": "",
        "rf": "\\unit{{\\watt\\per\\metre\\squared}}" if not normalize else "",
        "rf_sw": "\\unit{{\\watt\\per\\metre\\squared}}" if not normalize else "",
        "rf_lw": "\\unit{{\\watt\\per\\metre\\squared}}" if not normalize else "",
    }
    cbar_labels = {k: v for k, v in labels.items()}
    for k, v in units.items():
        if v != "":
            cbar_labels[k] = f"{cbar_labels[k]} ({v})"

    hist = histograms[category]
    if normalize:
        hist = hist / hist.max()
        peak = 1
    dims = list(hist.dims)
    values = [histograms[dim].values for dim in dims]

    peak_val = np.max(histograms[category]).squeeze() if peak is None else peak
    cm_obs = ListedColormap(
        plt.cm.get_cmap("Reds")(np.linspace(0, 0.8, histograms.attrs["n_repeats"]))
    )
    # cm_obs.set_over("darkred")
    plotting_kwargs = {
        "ob": {
            "cmap": cm_obs,
            "norm": Normalize(0, 1),
        },
        "oc": {"cmap": "Greens", "norm": Normalize(0, peak_val)},
        "rf": {
            "cmap": "coolwarm",
            "norm": (
                SymLogNorm(100, vmin=-1 * peak_val, vmax=peak_val)
                if not normalize
                else Normalize(-1, 1)
            ),
        },
    }
    hist_kwargs = plotting_kwargs[category[:2]] | hist_kwargs

    # remove overflow columns
    for i, dim in enumerate(dims):
        hist = hist.sel({dim: slice(values[i][1], values[i][-2])})
        values[i] = values[i][1:-1]

    if ax is None:
        ax = plt.gca()

    if len(dims) > 2:
        raise ValueError("Can only plot 2D histograms")

    handler = ax.pcolormesh(*values, hist.T, **hist_kwargs)

    if is_logspace(values[0]):
        ax.set_xscale("log")
    if is_logspace(values[1]):
        ax.set_yscale("log")

    if obs_contour:
        contour_observability(ax, histograms)
    if cbar:
        cb = plt.colorbar(
            handler,
            ax=ax,
            # fraction=0.048,
            # pad=0.04,
            # shrink=2.5,
            label=cbar_labels[category] if label_cbar else None,
        )
        cax = cb.ax
        if not label_cbar:
            ax.set_title(labels[category], loc="right", size="medium")
            cax.set_ylabel(units[category], ha="left", rotation=0)
            cax.yaxis.set_label_coords(1.5, 1.12)
        if category == "observability":
            cb.set_ticks([0, 0.5, 1])
        # cax = colorbar(handler, ax, label=cbar_labels[category])
    else:
        cax = None
    if tau_contour:
        contour_taus(ax, *dims, locator=LogLocator(numticks=10))

    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])

    return ax, cax


def contour_observability(ax, histograms, observability_threshold=0.5):
    observability = histograms["observability"]
    values = [histograms[dim].values for dim in list(observability.dims)]
    ax.contour(
        *values,
        observability.T,
        levels=[observability_threshold],
        colors="k",
        linewidths=1,
    )
    # ax.contourf(*values, observability.T, levels=[0.05, 0.95], colors="gray", alpha=0.2)
    return ax


def plot_observability_analyses(
    histograms,
    rf_var="rf",
    xlabel=None,
    ylabel=None,
    peaks=None,
    axes=None,
    colorbars=True,
    label_cbars=True,
    normalize=False,
):
    cats = ["observability", "occurrence", rf_var]
    peaks = (
        [np.max(np.abs(histograms[cat])).squeeze() for cat in cats]
        if peaks is None
        else peaks
    )
    # rf_peak = (
    #     np.max(np.abs(histograms[rf_var])).squeeze() if rf_peak is None else rf_peak
    # )
    # occ_peak = (
    #     np.max(histograms["occurrence"]).squeeze() if occ_peak is None else occ_peak
    # )
    cm_occ = plt.get_cmap("Reds")
    cm_occ.set_over("darkred")

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, category, peak in zip(axes, ["observability", "occurrence", rf_var], peaks):
        ax, cax = plot_observability_histogram(
            histograms,
            category,
            ax=ax,
            peak=peak,
            cbar=colorbars,
            label_cbar=label_cbars,
            normalize=False if category == "observability" else True,
        )

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    return axes


def plot_single_observability_analysis(
    ax, histogram, category, xlabel=None, ylabel=None, peak=None, label_cbar=True
):
    cbar_labels = {
        "observability": "Detection probability",
        "occurrence": "Contrail segments",
        "rf": "RF-weighted contrail segments",
    }

    peak = np.max(histogram).squeeze() if peak is None else peak
    cm = plt.get_cmap("Reds")
    cm.set_over("darkred")
    plotting_kwargs = {
        "observability": {
            "cmap": cm,
            "norm": Normalize(0, 3),
        },
        "occurrence": {"cmap": "Greens", "norm": Normalize(0, peak)},
        "rf": {
            "cmap": "coolwarm",
            "norm": SymLogNorm(100, vmin=-1 * peak, vmax=peak),
        },
    }

    ax, cax = plot_observability_histogram(
        histogram,
        category,
        ax=ax,
        hist_kwargs=plotting_kwargs[category],
        label_cbar=label_cbar,
    )
    contour_observability(ax, histogram)
    cax.set_ylabel(cbar_labels[category])

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax, cax


def annotate_observable_fraction(ax, histograms, category, observability_threshold=0.5):
    from simtrails.cocip.observability_histograms import observable_fraction

    obs, err = observable_fraction(histograms, category, observability_threshold)
    ax.text(
        0.05,
        0.95,
        f"Observable: {format_observability(obs, err)}",
        transform=ax.transAxes,
        backgroundcolor="w",
        zorder=999,
        ha="left",
        va="top",
    )
    return ax


def format_observability(obs_fraction, errs):
    return f"\\qty{{{obs_fraction*100:.1f}}}{{\\percent}} $+$\\qty{{{errs[0]*100:.1f}}}{{\\percent}} $-$\\qty{{{errs[1]*100:.1f}}}{{\\percent}}"


def annotate_observable_fractions(axes, histograms, observability_threshold=0.5):
    for ax, category in zip(axes[1:], ["occurrence", "rf"]):
        annotate_observable_fraction(ax, histograms, category, observability_threshold)
    return axes


def plot_observability_apportionment(
    observability_proportions,
    offset=0.0,
    scale=1.0,
    ax=None,
    # suffix="",
    hist_kwargs={},
    errbar_kwargs={},
    plot_errorbar=True,
    plot_bars=True,
    weight_vars=[None, "rf_lw", "rf_sw", "rf_net"],
    calc_pc=True,
):
    from matplotlib import pyplot as plt
    import seaborn as sns

    if calc_pc:
        stat = "density"
    else:
        stat = "count"
    if ax is None:
        fig, ax = plt.subplots()
    for i, w in enumerate(weight_vars):
        if w is None:
            w = "existent"
        if plot_bars:
            sns.histplot(  # showish (not hugely)
                observability_proportions[w]
                .loc[OBS_HUE_ORDER]
                .to_frame()
                .reset_index(names=["hue"]),
                y=[len(weight_vars) - i - 1 + offset * 0.8] * len(OBS_HUE_ORDER),
                multiple="stack",
                stat=stat,
                bins=1,
                ax=ax,
                shrink=0.8 * scale,
                hue="hue",
                weights=w,
                hue_order=OBS_HUE_ORDER,
                palette=[
                    sns.color_palette()[0],
                    (*sns.color_palette()[1], 0.3),
                    (*sns.color_palette()[2], 0.3),
                ],  # first 3 colors from seaborn palette; with 2nd and 3rd with alpha=0.3
                **hist_kwargs,
            )

        if plot_errorbar:
            obs_pc = observability_proportions[w]["observable"]
            obs_pc_low = observability_proportions[w]["observable_low"]
            obs_pc_high = observability_proportions[w]["observable_high"]

            errbar_kwargs = (
                dict(
                    fmt="o",
                    color="k",
                    markersize=3,
                    capsize=3,
                    markeredgecolor="k",
                )
                | errbar_kwargs
            )
            ax.errorbar(
                x=obs_pc,
                y=len(weight_vars) - i - 1 + offset * 0.8,
                xerr=[[obs_pc - obs_pc_low], [obs_pc_high - obs_pc]],
                **errbar_kwargs,
            )

            print(
                f"\t{w}: {obs_pc:.2%} +{obs_pc_high-obs_pc:.2%} -{obs_pc-obs_pc_low:.2%}"
            )

    # make "too narrow" and "too optically thin" partially transparent
    for patch in ax.patches:
        if (patch.get_x() > 0) and patch.fill and (patch.get_facecolor()[-1] != 0.0):
            patch.set_alpha(0.3)
    return ax
