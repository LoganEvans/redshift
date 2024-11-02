from pprint import pprint
import json
import os
import csv
import numpy as np
import scipy
import pathlib

from matplotlib import pyplot as plt
from dataclasses import dataclass, field

DATA_DIR = (pathlib.Path(os.environ["VIRTUAL_ENV"]).parent / "data").resolve()

# Using 2021J. Distance: 18.110 Mpsc. Redshift: 0.0024. Magnitude: 12.5
SN2021J_distance = 18.110
SN2021J_redshift = 0.0024
SN2021J_magnitude = 12.5


@dataclass()
class Supernova:
    redshift: float
    name: str | None = field(default=None)
    magnitude: float | None = field(default=None)
    stretch: float | None = field(default=None)
    color: float | None = field(default=None)

    @staticmethod
    def from_wolfram():
        """https://datarepository.wolframcloud.com/resources/Type-Ia-Supernova-Data/"""
        # https://arxiv.org/pdf/1401.4064

        data = []

        with open(DATA_DIR / "type-Ia-supernova-data.json", "r") as json_file:
            raw = json.load(json_file)[1]
            for item in raw[1:]:
                fields = {}

                for rule in item[1:]:
                    if rule[1] == "'supernova name'":
                        fields["name"] = rule[2].strip("'")
                    else:
                        fields[rule[1].strip("'")] = rule[2]

                data.append(Supernova(**fields))

        return data

    @property
    def z(self):
        return self.redshift

    @property
    def uncorrected_uncalibrated_distance(self):
        return 100 ** (self.magnitude / 10)

    @property
    def corrected_uncalibrated_distance(self):
        return (100 ** (self.magnitude / 10)) / np.sqrt((self.z + 1))

    def distance_mpsc(self, slope, intercept):
        return (
            SN2021J_distance
            * (slope * self.z + intercept)
            / (slope * SN2021J_redshift + intercept)
        )

    @property
    def velocity_kms(self):
        c = 2.99792458e5  # km/s
        return self.z * c


if __name__ == "__main__":
    data = Supernova.from_wolfram()

    xs = [sn.redshift for sn in data]

    ys = [sn.uncorrected_uncalibrated_distance for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="red")

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    uncorrected_siegel_legend = (
        f"Siegel best fit: d = {coefficients.slope:.1f}z + {coefficients.intercept:.1f}"
    )
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.intercept + coefficients.slope),
        color="darkred",
        linewidth=0.5,
    )

    ys = [sn.corrected_uncalibrated_distance for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="blue")

    plt.xlabel("z")
    plt.ylabel("unscaled d")

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    corrected_siegel_legend = (
        f"Siegel best fit: d = {coefficients.slope:.1f}z + {coefficients.intercept:.1f}"
    )
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.intercept + coefficients.slope),
        color="darkblue",
        linewidth=0.5,
    )

    plt.title("Unscaled Distance vs Redshift")
    plt.legend(
        [
            "luminosity uncorrected for redshift",
            uncorrected_siegel_legend,
            "luminosity corrected for redshift",
            corrected_siegel_legend,
        ]
    )

    xs = [sn.distance_mpsc(coefficients.slope, coefficients.intercept) for sn in data]
    ys = [sn.velocity_kms for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    print("km/s per Mpsc (note: dependent on a single datapoint)")
    print(coefficients)

    plt.show()
