# %%
import numpy as np

# v1: Rotated-rectangle contrail
# v2: Gaussian contrail
# v3: Revised lookup table; includes contrail length record; "optical_thickness_width" at higher-resolution
# v4: Widths are now log-spaced, with more bins
# v5: Remove "edge detections" on really wide contrails by filtering on orientation
# v6: Bring down the lower limit ofr eff radius to 0.1 um
# v7: Use width at 1/e of peak value (Schumann 2017 - cocip specification)
# v8: With adjusted detection algorithm (no regional gradient filtering) + bg cirrus introduction
# v9: Thicker background cirrus, and NEdT moved to 0.03 K (from 0.15 K) (!!), and moved Mannstein BT difference threshold to 0.2 K
# v10: Move Mannstein config back to Google version, added additional OGAD config for special case # big observability drop, potentially due to calibration errors or Google's minimum length and area requirements

saved_sensitivities = {
    "optical_thickness": {
        "properties": {
            "iwp": np.logspace(-3, 1.5, 41),  # quicker
            "eff_radius": np.logspace(np.log10(0.1), np.log10(50), 40),  # quicker
            "resolution": np.array([0.5, 2.0]),
        },
        "contrail_parameters": {"width": 2},
        "class": "GaussianContrail",
        "version": 7,
    },
    "optical_thickness_width": {
        "properties": {
            "iwp": np.logspace(-3, 1.5, 41),
            "eff_radius": np.logspace(np.log10(0.1), np.log10(50), 40),
            "width": np.logspace(np.log10(0.025), np.log10(25), 30),
            "resolution": np.array([0.5, 2.0, 7.0]),
        },
        "class": "GaussianContrail",
        "version": 11,
    },
    "bg_cirrus_optical_thickness_width": {
        "properties": {
            "iwp": np.logspace(-3, 1.5, 41),
            "eff_radius": np.logspace(np.log10(0.1), np.log10(50), 40),
            "width": np.logspace(np.log10(0.025), np.log10(25), 30),
            "resolution": np.array([0.5, 2.0]),
        },
        "contrail_properties": {"background": "8km_cirrus"},
        "class": "GaussianContrail",
        "version": 11,
    },
    "bg_cirrus_optical_thickness_width_bigpcles": {
        "properties": {
            "iwp": np.logspace(-3, 1.5, 41),
            "eff_radius": np.logspace(np.log10(5), np.log10(50), 40),
            "width": np.logspace(np.log10(0.025), np.log10(25), 30),
            "resolution": np.array([0.5, 2.0]),
        },
        "contrail_properties": {"background": "8km_cirrus_bigpcles"},
        "class": "GaussianContrail",
        "version": 1,
    },
    "optical_thickness_width_ogadconfig": {
        "properties": {
            "iwp": np.logspace(-3, 1.5, 41),
            "eff_radius": np.logspace(np.log10(0.1), np.log10(50), 40),
            "width": np.logspace(np.log10(0.025), np.log10(25), 30),
            "resolution": np.array([0.5, 2.0]),
        },
        "cda_properties": {"config": "ogad"},
        "class": "GaussianContrail",
        "version": 10,
    },
    "coarse_optical_thickness_width": {
        "properties": {
            "iwp": np.logspace(-3, 1.5, 11),
            "eff_radius": np.logspace(np.log10(0.1), np.log10(50), 11),
            "width": np.logspace(np.log10(0.025), np.log10(25), 11),
            "resolution": np.array([0.5, 2.0]),
        },
        "class": "GaussianContrail",
        "version": 7,
    },
}


sensitivity_values = {
    "length": np.array(np.logspace(0, 7, 8, base=2)),
    "iwc": np.concatenate(
        [np.linspace(0.0005, 0.0025, 5), np.linspace(0.003, 0.02, 18)]
    ),
    "angle": np.linspace(0, 90, 10),
    "width": np.linspace(0.5, 5, 10),
    "altitude": np.linspace(8, 12.5, 10),
    "resolution": np.array([0.25, 0.5, 1, 2]),
    # "OPTICAL_DEPTH": None,
    "eff_radius": np.linspace(5, 40, 8),
    "depth": np.linspace(0.2, 3, 15),
    "hour": np.array([0, 6, 12, 18]),
    "atmosphere_file": np.array(["us", "t", "ms", "mw", "ss", "sw"]),
    "iwp": np.linspace(0, 120, 13),
    "n_ice": np.linspace(5e6, 1e8, 20),
}


if __name__ == "__main__":
    from simtrails.sensitivity_result import SensitivityResult

    for sensitvity in saved_sensitivities:
        print(f"Running sensitivity: {sensitvity}")
        SensitivityResult.from_name(sensitvity, n_proc=128)
        print(f"Finished sensitivity: {sensitvity}\n")

# %%
