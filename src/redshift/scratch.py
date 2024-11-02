import json
import os
import csv
import numpy as np
import scipy
import pathlib
from typing import ClassVar
from matplotlib import pyplot as plt
from dataclasses import dataclass, field

REPO_DIR = pathlib.Path(os.environ["VIRTUAL_ENV"]).parent.resolve()
DATA_DIR = REPO_DIR / "data"
GRAPHS_DIR = REPO_DIR / "graphs"


@dataclass()
class Supernova:
    redshift: float
    name: str | None = field(default=None)
    magnitude: float | None = field(default=None)
    stretch: float | None = field(default=None)
    color: float | None = field(default=None)

    H0: ClassVar[float] = 70

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

    @staticmethod
    def _uncorrected_uncalibrated_distance(magnitude):
        return 10 ** (magnitude / 5)

    @property
    def uncorrected_uncalibrated_distance(self):
        return self._uncorrected_uncalibrated_distance(self.magnitude)

    def uncorrected_calibrated_distance(self, slope, intercept):
        return (self.uncorrected_uncalibrated_distance - intercept) * self.H0 / slope

    @staticmethod
    def _corrected_uncalibrated_distance(magnitude, z):
        return (10 ** (magnitude / 5)) / np.sqrt(z + 1)

    @property
    def corrected_uncalibrated_distance(self):
        return self._corrected_uncalibrated_distance(self.magnitude, self.z)

    def corrected_calibrated_distance(self, slope, intercept):
        return (self.corrected_uncalibrated_distance - intercept) * self.H0 / slope

    @staticmethod
    def _velocity_kms(z):
        c = 2.99792458e5  # km/s
        return z * c

    @property
    def velocity_kms(self):
        return self._velocity_kms(self.z)


if __name__ == "__main__":
    data = Supernova.from_wolfram()

    def uncorrected_uncalibrated_graph():
        plt.cla()
        xs = [sn.redshift for sn in data]
        ys = [sn.uncorrected_uncalibrated_distance for sn in data]
        plt.scatter(xs, ys, s=2, marker=".", color="red")

        coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
        plt.axline(
            (0, coefficients.intercept),
            (1, coefficients.slope + coefficients.intercept),
            color="black",
            linewidth=0.5,
        )

        plt.title("Unscaled Uncorrected Distance vs Redshift")
        plt.legend(
            [
                "Supernova",
                f"Siegel best fit: d = {coefficients.slope:.2f}z + {coefficients.intercept:.2f}",
            ]
        )
        plt.xlabel("z")
        plt.ylabel("unscaled d")
        plt.savefig(GRAPHS_DIR / "uncorrected_uncalibrated.png", bbox_inches="tight")

    uncorrected_uncalibrated_graph()

    def corrected_uncalibrated_graph():
        plt.cla()
        xs = [sn.redshift for sn in data]
        ys = [sn.corrected_uncalibrated_distance for sn in data]
        plt.scatter(xs, ys, s=2, marker=".", color="blue")

        coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
        plt.axline(
            (0, coefficients.intercept),
            (1, coefficients.slope + coefficients.intercept),
            color="black",
            linewidth=0.5,
        )

        plt.title("Unscaled Corrected Distance vs Redshift")
        plt.legend(
            [
                "Supernova",
                f"Siegel best fit: d = {coefficients.slope:.2f}z + {coefficients.intercept:.2f}",
            ]
        )
        plt.xlabel("z")
        plt.ylabel("unscaled d")
        plt.savefig(GRAPHS_DIR / "corrected_uncalibrated.png", bbox_inches="tight")

    corrected_uncalibrated_graph()

    def both_calibrated_graph():
        plt.cla()
        xs = [sn.redshift for sn in data]

        ys = [sn.uncorrected_uncalibrated_distance for sn in data]
        coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
        ys = [
            sn.uncorrected_calibrated_distance(
                coefficients.slope, coefficients.intercept
            )
            for sn in data
        ]
        plt.scatter(xs, ys, s=2, marker=".", color="red")

        ys = [sn.corrected_uncalibrated_distance for sn in data]
        coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
        ys = [
            sn.corrected_calibrated_distance(coefficients.slope, coefficients.intercept)
            for sn in data
        ]
        plt.scatter(xs, ys, s=2, marker=".", color="blue")

        plt.axline(
            (0, 0),
            (1, Supernova.H0),
            color="black",
            linewidth=0.5,
        )

        plt.xlabel("z")
        plt.ylabel("Mpsc")

        plt.title("Scaled Distance vs Redshift")
        plt.legend(["uncorrected", "corrected", f"H0 = {Supernova.H0:.0f} km/s / Mpsc"])
        plt.savefig(GRAPHS_DIR / "both_calibrated.png", bbox_inches="tight")

    both_calibrated_graph()

    #plt.show()
