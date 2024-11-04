import numpy as np


class Atmosphere:
    columns = [
        "z",
        "p",
        "T",
        "air",
        "ozone",
        "o2",
        "h2o",
        "co2",
        "no2",
    ]
    columns = {k: i for i, k in enumerate(columns)}

    def __init__(self, afgl_file):
        self.afgl_file = afgl_file  # .dat file

        self.data = np.loadtxt(afgl_file)

    def value_at_altitude(self, variable, z):
        return np.interp(z, self.data[::-1, 0], self.data[::-1, self.columns[variable]])

    @classmethod
    def from_name(cls, name):
        """
        name: str
            Name of the atmosphere, i.e. us (U.S. Standard), ms (Midlatitudes Summer), mw (midlatitudes winter), ss (subarctic summer), sw (subarctic winter), or t (tropics).
        """
        from pathlib import Path

        return cls(Path(__file__).parent / "data" / "atmospheres" / f"afgl{name}.dat")
