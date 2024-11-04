from __future__ import annotations
from dataclasses import dataclass, field, asdict
from numpy.polynomial.polynomial import Polynomial
from datetime import date
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from pprint import pprint
from typing import ClassVar
import csv
import jdcal
import json
import numpy as np
import os
import pathlib
import random
import scipy

REPO_DIR = pathlib.Path(os.environ["VIRTUAL_ENV"]).parent.resolve()
DATA_DIR = REPO_DIR / "data"
GRAPHS_DIR = REPO_DIR / "graphs"


def parse_jd_date(jd: str):
    year, month, day, _ = jdcal.jd2gcal(float(jd), 0)
    return date(year=year, month=month, day=day)


@dataclass()
class Supernova:
    name: str | None
    magnitude: float
    magnitude_error: float
    redshift: float

    H0: ClassVar[float] = 70

    @property
    def z(self):
        return self.redshift

    @property
    def corrected_magnitude(self):
        return self.magnitude * (self.z + 1)

    @staticmethod
    def _uncalibrated_distance(corrected, magnitude, z):
        d = 10 ** (magnitude / 5)
        if corrected:
            d /= np.sqrt(z + 1)
        return d

    def uncalibrated_distance(self, corrected):
        return self._uncalibrated_distance(corrected, self.magnitude, self.z)

    def calibrated_distance(self, calibration: Calibration):
        return calibration.correction * self.uncalibrated_distance(
            corrected=calibration.corrected
        )

    @property
    def velocity_kms(self):
        c = 299792.458  # km/s
        return self.z * c

    @staticmethod
    def from_perlmutter():
        # https://iopscience.iop.org/article/10.1086/307221/pdf
        data = []

        with open(DATA_DIR / "perlmutter.csv", "r") as csv_file:
            raw = csv.reader(csv_file)
            header = next(raw)
            idx = {val: i for i, val in enumerate(header)}

            for row in raw:
                sn = Supernova(
                    name=row[idx["name"]],
                    magnitude=float(row[idx["mb_eff"]]),
                    magnitude_error=float(row[idx["mb_eff_error"]]),
                    redshift=float(row[idx["z"]])
                )
                data.append(sn)
        return data

    @staticmethod
    def from_betoule():
        # https://arxiv.org/pdf/1401.4064
        data = []

        name_idx = 0
        z_idx = 1
        magnitude_idx = 4
        magnitude_error_idx = 5

        with open(DATA_DIR / "jla_lcparams.txt", "r") as csv_file:
            raw = csv.reader(csv_file)
            header = next(raw)
            for row in raw:
                sn = Supernova(
                    name=row[name_idx],
                    magnitude=float(row[magnitude_idx]),
                    magnitude_error=float(row[magnitude_error_idx]),
                    redshift=float(row[z_idx]),
                )
                data.append(sn)
        return data

    @staticmethod
    def from_rochester():
        supernovas = []
        with open(DATA_DIR / "rochester-data.csv", "r") as csv_file:
            reader = csv.reader(csv_file)

            header = next(reader)
            idx = {field: i for i, field in enumerate(header)}

            for row in reader:
                if row[idx["Type"]] != "Ia":
                    continue

                z = row[idx["z"]]
                if z == "n/a":
                    continue
                z = float(z)

                magnitude = float(row[idx["Max mag"]])
                if magnitude <= 9.9 or magnitude >= 99.9:
                    continue

                if row[idx["Discoverer(s)"]] == "Supernova Legacy Project":
                    continue

                supernovas.append(
                    Supernova(name=row[idx["SN"]], magnitude=magnitude, redshift=z)
                )
        return supernovas


class Calibration:
    def __init__(self, data, corrected):
        self.corrected = corrected

        # distance = [sn.uncalibrated_distance(corrected) for sn in data]
        # velocity = [sn.velocity_kms for sn in data]

        # coefficients = scipy.stats.siegelslopes(x=distance, y=velocity)
        # print(coefficients)
        # self.intercept = coefficients.intercept

        # c = 299792.458  # km/s
        ks = []

        # d(z) = ud(z) * k
        # H0 = v(z) / d(z)
        # H0 = v(z) / (ud(z) * k)
        # ud(z) * k = v(z) / H0
        # k = v(z) / (H0 * ud(z))

        for sn in data:
            k = sn.velocity_kms / (
                Supernova.H0 * sn.uncalibrated_distance(corrected=corrected)
            )
            # print(k, sn.velocity_kms / (sn.uncalibrated_distance(corrected=corrected) * k))
            ks.append(k)

        self.correction = np.median(ks)


def uncalibrated_graph(data, corrected=True, save=True):
    plt.cla()
    xs = [sn.redshift for sn in data]
    ys = [sn.uncalibrated_distance(corrected) for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="blue" if corrected else "red")

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
    )

    plt.title(
        f"Unscaled {'Corrected' if corrected else 'Uncorrected'} Distance vs Redshift"
    )
    plt.legend(
        [
            "Supernova",
            f"Siegel best fit: d = {coefficients.slope:.2f}z + {coefficients.intercept:.2f}",
        ]
    )
    plt.xlabel("z")
    plt.ylabel("unscaled d")
    if save:
        plt.savefig(
            GRAPHS_DIR
            / f"{'corrected' if corrected else 'uncorrected'}_uncalibrated.png",
            bbox_inches="tight",
        )
        plt.cla()


def both_calibrated_redshift_vs_distance_graph(data, save=True):
    plt.cla()
    ys = [sn.redshift for sn in data]

    cal = Calibration(data, corrected=False)
    xs = [sn.calibrated_distance(cal) for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="red")

    cal = Calibration(data, corrected=True)
    xs = [sn.calibrated_distance(cal) for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="blue")

    plt.xlabel("distance (Mpsc)")
    plt.ylabel("z")

    plt.title("Scaled Distance vs Redshift")
    plt.legend(["uncorrected", "corrected", f"H0 = {Supernova.H0:.0f} km/s / Mpsc"])
    if save:
        plt.savefig(
            GRAPHS_DIR / "both_calibrated_distance_vs_redshift.png", bbox_inches="tight"
        )
        plt.cla()


def both_calibrated_velocity_vs_distance_graph(data, save=True):
    plt.cla()
    ys = [sn.velocity_kms for sn in data]

    cal = Calibration(data, corrected=False)
    xs = [sn.calibrated_distance(cal) for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="red", label="uncorrected")

    cal = Calibration(data, corrected=True)
    xs = [sn.calibrated_distance(cal) for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="blue", label="corrected")

    plt.axline(
        (0, 0),
        (1, Supernova.H0),
        color="black",
        linewidth=0.5,
        label=f"H0 = {Supernova.H0:.0f} km/s / Mpsc",
    )

    plt.xlabel("distance (Mpsc)")
    plt.ylabel("velocity (km/s)")

    plt.title("Velocity vs Distance")
    plt.legend()
    if save:
        plt.savefig(
            GRAPHS_DIR / "both_calibrated_velocity_vs_distance.png", bbox_inches="tight"
        )
        plt.cla()


def bootstrap_z_vs_ud(data, corrected, trials):
    slopes = []
    for i in range(trials):
        print(
            f"Bootstraping trial z vs ud (corrected={corrected}): {i:7.0f}/{trials}",
            end="\r",
        )
        resampled = [Supernova(**asdict(sn)) for sn in np.random.choice(data, size=len(data), replace=True)]
        for sn in resampled:
            sn.magnitude += scipy.stats.norm.rvs(scale=sn.magnitude_error)

        coefficients = scipy.stats.siegelslopes(
            x=[sn.uncalibrated_distance(corrected=corrected) for sn in resampled],
            y=[sn.redshift for sn in resampled],
        )
        slopes.append(coefficients.slope)

    print()
    return slopes


def bootstrap_z_vs_ud_graph(data, corrected, trials, save=True):
    med = np.median([sn.z for sn in data])

    left = [sn for sn in data if sn.z <= med]
    right = [sn for sn in data if sn.z > med]

    left_z_vs_ud = bootstrap_z_vs_ud(data=left, corrected=corrected, trials=trials)
    right_z_vs_ud = bootstrap_z_vs_ud(data=right, corrected=corrected, trials=trials)

    bins = np.linspace(
        min(min(left_z_vs_ud), min(right_z_vs_ud)),
        max(max(left_z_vs_ud), max(right_z_vs_ud)),
        100,
    )

    plt.hist(
        left_z_vs_ud,
        bins,
        alpha=0.5,
        label=f"z vs ud bootstrapped from z <= {med:.2f}\n(median: {np.median(left_z_vs_ud):.2f})",
    )

    plt.hist(
        right_z_vs_ud,
        bins,
        alpha=0.5,
        label=f"z vs ud bootstrapped from z > {med:.2f}\n(median: {np.median(right_z_vs_ud):.2f})",
    )

    plt.legend()
    plt.title("Bootstrapped z vs ud slope")
    if save:
        plt.savefig(
            GRAPHS_DIR / f"bootstrapped_H0_{'' if corrected else 'un'}corrected.png",
            bbox_inches="tight",
        )
        plt.cla()


def bootstrap_H0(data, corrected, trials, save=True):
    slopes = []
    for i in range(trials):
        print(
            f"Bootstraping H0 (corrected={corrected}): {i:7.0f}/{trials}",
            end="\r",
        )

        resampled = [Supernova(**asdict(sn)) for sn in np.random.choice(data, size=len(data), replace=True)]
        for sn in resampled:
            sn.magnitude += scipy.stats.norm.rvs(scale=sn.magnitude_error)

        cal = Calibration(resampled, corrected=corrected)
        xs = [sn.calibrated_distance(cal) for sn in resampled]
        ys = [sn.velocity_kms for sn in resampled]
        coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
        slopes.append(coefficients.slope)
    print()
    return slopes


def bootstrap_H0_graph(data, corrected, trials, save=True):
    med = np.median([sn.z for sn in data])
    #z = 0.15

    left = [sn for sn in data if sn.z <= med]
    right = [sn for sn in data if sn.z > med]

    left_H0 = bootstrap_H0(data=left, corrected=corrected, trials=trials)
    right_H0 = bootstrap_H0(data=right, corrected=corrected, trials=trials)

    bins = np.linspace(
        min(min(left_H0), min(right_H0)),
        max(max(left_H0), max(right_H0)),
        100,
    )

    plt.hist(
        left_H0,
        bins,
        alpha=0.5,
        label=f"H0 bootstrapped from z <= {med:.2f}\n(median: {np.median(left_H0):.2f})",
    )

    plt.hist(
        right_H0,
        bins,
        alpha=0.5,
        label=f"H0 bootstrapped from z > {med:.2f}\n(median: {np.median(right_H0):.2f})",
    )

    plt.legend()
    plt.title(f"Bootstrapped H0 calibrated at H0 = {Supernova.H0:.0f} ({'corrected' if corrected else 'uncorrected'})")
    if save:
        plt.savefig(
            GRAPHS_DIR / f"bootstrapped_H0_{'' if corrected else 'un'}corrected.png",
            bbox_inches="tight",
        )
        plt.cla()


def generate_all_graphs(data, save=True):
    uncalibrated_graph(data, corrected=True, save=save)
    uncalibrated_graph(data, corrected=False, save=save)
    both_calibrated_redshift_vs_distance_graph(data, save=save)
    both_calibrated_velocity_vs_distance_graph(data, save=save)
    bootstrap_H0_graph(data, corrected=True, trials=1000, save=save)
    bootstrap_H0_graph(data, corrected=False, trials=1000, save=save)
    bootstrap_z_vs_ud_graph(data, corrected=True, trials=1000, save=save)


if __name__ == "__main__":
    # data = Supernova.from_rochester()
    data = Supernova.from_betoule()
    generate_all_graphs(data)
    exit()
    #data = Supernova.from_perlmutter()

    xs = [sn.redshift for sn in data]
    ys = [sn.uncalibrated_distance(corrected=True) for sn in data]
    plt.scatter(xs, ys, s=2, marker=".", color="blue", label="corrected")
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
    )

    #poly = Polynomial(np.polyfit(xs, ys, 2)[::-1])
    #poly_xs, poly_ys = poly.linspace()
    #plt.plot(poly_xs, poly_ys, lw=1, color="red")

    plt.show()
    exit()

    both_calibrated_redshift_vs_distance_graph(data, save=False)
    #both_calibrated_velocity_vs_distance_graph(data, save=False)

    #bootstrap_H0_graph(data, corrected=False, trials=1000, save=False)
    #bootstrap_z_vs_ud_graph(data, corrected=True, trials=1000, save=False)

    plt.show()
