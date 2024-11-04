import numpy as np
import logging


def is_logspace(x):
    diffs = np.diff(np.log10(x))

    return np.abs(np.diff(diffs)).max() < 1e-6 * np.abs(diffs).min()


def histogram(df, columns, bins=100, weights=None):
    sample = df[columns].to_numpy()
    if weights is not None:
        weights = df[weights].to_numpy()
    hist, edges = np.histogramdd(sample, bins=bins, weights=weights)

    expected_total = weights.sum() if weights is not None else len(df)
    pc_included = hist.sum() / expected_total

    if pc_included < 0.99:
        logging.warning(
            f"Only {pc_included:.2%} of the cocip {columns} data is included in the histogram"
        )

    return hist, edges


def bin_edges_from_midpoints(midpoints, log=False):
    if log:
        midpoints = np.log10(midpoints)
    edges = np.zeros(len(midpoints) + 1)
    edges[1:-1] = (midpoints[1:] + midpoints[:-1]) / 2
    edges[0] = midpoints[0] - (midpoints[1] - midpoints[0]) / 2
    edges[-1] = midpoints[-1] + (midpoints[-1] - midpoints[-2]) / 2
    if log:
        edges = 10**edges
    return edges


def observability_analysis(
    bin_edges: np.ndarray,
    cocip_data,
    splitting_function=lambda df, low, high: df[df.age.between(low, high)],
    resolution=2,
    rf=False,
):
    import logging
    from simtrails.cocip.observability_histograms import observable_fraction

    pcs_obs = []
    errs_obs = []

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        histograms = cocip_data.observability_dataset(
            "optical_thickness_width",
            resolution,
            df_subset_function=lambda df: splitting_function(df, low, high),
            join_contrails=True,
        )
        pc_obs, err_obs = observable_fraction(
            histograms,
            "occurrence" if not rf else "rf",
            observability_threshold=0.5,
        )

        pcs_obs.append(pc_obs)
        errs_obs.append(err_obs)

        logging.debug(f"\tBin: {low:.1f} - {high:.1f}")
        logging.debug(
            f"\tObservability: {pc_obs:.1%} +{err_obs[0]:.1%} - {err_obs[1]:.1%}"
        )

    errs_obs = np.array(errs_obs).T

    return pcs_obs, errs_obs
