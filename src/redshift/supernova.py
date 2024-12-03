from __future__ import annotations
from bisect import bisect
from dataclasses import dataclass, field, asdict
from datetime import date
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from pprint import pprint
from typing import ClassVar
from enum import Enum
import csv
import jdcal
import json
import math
import numpy as np
import os
import pathlib
import random
import scipy

REPO_DIR = pathlib.Path(os.environ["VIRTUAL_ENV"]).parent.resolve()
DATA_DIR = REPO_DIR / "data"
GRAPHS_DIR = REPO_DIR / "graphs"


class Correction(Enum):
    NONE = 0
    ONE = 1  # Single correction factor: 1+z
    TWO = 2  # Two correction factors: (1+z)^2


def parse_jd_date(jd: str):
    year, month, day, _ = jdcal.jd2gcal(float(jd), 0)
    return date(year=year, month=month, day=day)


@dataclass()
class Supernova:
    redshift: float
    name: str | None
    magnitude: float

    magnitude_error: float | None = field(default=None)
    magnitude_uncorrected: float | None = field(default=None)
    x1: float | None = field(default=None)
    color: float | None = field(default=None)
    absolute_magnitude: float | None = field(default=None)

    MU: float | None = field(default=None)

    H0: ClassVar[float] = 70

    @property
    def z(self):
        return self.redshift

    def corrected_magnitude(self, correction=Correction.NONE):
        if correction == Correction.NONE:
            return self.magnitude
        elif correction == Correction.TWO:
            return self.magnitude - 2 * np.log(1 + self.z) / np.log(100**0.2)
        else:  # Tired light
            return self.magnitude - np.log(1 + self.z) / np.log(100**0.2)

    def inflation_rate(self):
        # A = P e^(rt)
        # T = xA
        # A/P = e^(rx A)
        # ln(A/P) = rx A
        # r = ln(A/P) / (Ax)
        A = self.mu_distance(correction=Correction.ONE)
        P = self.orig_distance(correction=Correction.ONE)
        x = 3.26156
        return math.log(A / P) / (x * A)

    @property
    def velocity_kms(self):
        c = 299792.458  # km/s

        z2 = self.z * self.z

        return (c * z2 + 2 * c * self.z) / (z2 + 2 * self.z + 2)

    def mu(self, correction=Correction.NONE):
        return self.corrected_magnitude(correction) - self.absolute_magnitude

    def mu_distance(self, correction=Correction.ONE, use_mpc=True):
        d = 10 ** (1 + self.mu(correction) / 5)
        if use_mpc:
            d /= 1e6
        return d

    def orig_distance(self, correction=Correction.ONE):
        return self.mu_distance(correction) / (self.z + 1)

    def delta_distance(self, correction=Correction.ONE):
        return self.mu_distance(correction) - self.orig_distance(correction)

    def energy_loss(self):
        # E = hc / l
        # E_loss = hc/l - hc/(l*(1+z))
        # E_loss = (hc(1+z) - hc) / (l(1+z))
        # E_loss = hcz / (l(1+z))
        # E_loss = hc/l * (z / (1+z))
        return self.z / (1 + self.z)

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
                    redshift=float(row[idx["z"]]),
                )
                data.append(sn)
        return data

    @staticmethod
    def from_betoule():
        # https://arxiv.org/pdf/1401.4064
        data = []

        with open(DATA_DIR / "jla_lcparams.txt", "r") as csv_file:
            raw = csv.reader(csv_file)
            header = next(raw)
            idx = {val: i for i, val in enumerate(header)}
            for row in raw:
                sn = Supernova(
                    name=row[idx["#name"]],
                    magnitude=float(row[idx["mb"]]),
                    magnitude_error=float(row[idx["dmb"]]),
                    redshift=float(row[idx["zcmb"]]),
                    x1=float(row[idx["x1"]]),
                    color=float(row[idx["color"]]),
                    absolute_magnitude=float(row[idx["3rdvar"]]),
                )
                data.append(sn)
        return data

    @staticmethod
    def from_abbott():
        # https://github.com/des-science/DES-SN5YR/blob/main/4_DISTANCES_COVMAT/DES-SN5YR_HD%2BMetaData.csv
        data = []

        with open(DATA_DIR / "DES-SN5YR_HD+MetaData.csv", "r") as csv_file:
            raw = csv.reader(csv_file)
            header = next(raw)
            idx = {val: i for i, val in enumerate(header)}
            for row in raw:
                sn = Supernova(
                    name=row[idx["CID"]],
                    magnitude=float(row[idx["mB_corr"]]),
                    magnitude_error=float(row[idx["mBERR"]]),
                    magnitude_uncorrected=float(row[idx["mB"]]),
                    redshift=float(row[idx["zHD"]]),
                    x1=float(row[idx["x1"]]),
                    MU=float(row[idx["MU"]]),
                    # absolute_magnitude=-19.102746,
                    absolute_magnitude=-19.3232,
                    color=float(row[idx["c"]]),
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


def reduce_data(data, step=0.002) -> list[Supernova]:
    old_data = sorted(data, key=lambda sn: sn.z)
    new_data_z = set()
    new_data = []

    z = 0

    while True:
        i = bisect(old_data, z, key=lambda sn: sn.z)
        if i >= len(old_data):
            break
        sn = old_data[i]
        if sn.z not in new_data_z:
            new_data_z.add(sn.z)
            new_data.append(sn)
        z = z + step

    return new_data


def all_mu_distance_vs_redshift_graph(data, save=True):
    plt.rcParams["figure.figsize"] = (8, 6)

    reduced_data = reduce_data(data)
    xs = [sn.z for sn in reduced_data]

    ys = [sn.mu_distance(correction=Correction.ONE) for sn in reduced_data]
    plt.scatter(
        xs,
        ys,
        s=15,
        marker="^",
        linewidths=0.7,
        facecolors="none",
        edgecolors="blue",
        label="k(z) = 1+z",
    )
    coefficients = scipy.stats.siegelslopes(
        x=[sn.z for sn in data],
        y=[sn.mu_distance(correction=Correction.ONE) for sn in data],
    )
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    ys = [sn.mu_distance(correction=Correction.NONE) for sn in reduced_data]
    plt.scatter(xs, ys, s=15, marker="x", linewidths=0.7, color="red", label="k(z) = 1")
    coefficients = scipy.stats.siegelslopes(
        x=[sn.z for sn in data],
        y=[sn.mu_distance(correction=Correction.NONE) for sn in data],
    )
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    # ys = [sn.mu_distance(correction=Correction.TWO) for sn in data]
    # plt.scatter(
    #    xs,
    #    ys,
    #    s=15,
    #    marker="o",
    #    linewidths=0.7,
    #    facecolors="none",
    #    edgecolors="green",
    #    label="k(z) = (1+z)^2",
    # )
    # coefficients = scipy.stats.siegelslopes(
    #    x=[sn.z for sn in old_data],
    #    y=[sn.mu_distance(correction=Correction.TWO) for sn in old_data],
    # )
    # plt.axline(
    #    (0, coefficients.intercept),
    #    (1, coefficients.slope + coefficients.intercept),
    #    color="black",
    #    linewidth=0.5,
    #    linestyle="--",
    # )

    plt.xlabel("z")
    plt.ylabel("distance (Mps)")

    plt.title("Distance vs Redshift")
    plt.legend()  # ["uncorrected", "corrected", f"H0 = {Supernova.H0:.0f} km/s / Mpsc"])
    if save:
        plt.savefig(GRAPHS_DIR / "mu_distance_vs_redshift.png", bbox_inches="tight")
        plt.cla()


def hubble_diagram_graph(data, save=True):
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.xscale("log")

    reduced_data = reduce_data(data, step=0.003)

    xs = [sn.redshift for sn in reduced_data]
    ys = [sn.mu(correction=Correction.ONE) for sn in reduced_data]
    plt.scatter(
        xs,
        ys,
        s=15,
        marker="x",
        color="red",
        linewidths=0.7,
        label="k(z) = 1",
    )

    xs = [sn.redshift for sn in reduced_data]
    ys = [sn.mu(correction=Correction.NONE) for sn in reduced_data]
    plt.scatter(
        xs,
        ys,
        s=15,
        marker="^",
        facecolors="none",
        edgecolors="blue",
        linewidths=0.7,
        label="k(z) = 1+z",
    )

    plt.title("Hubble diagram")
    plt.xlabel("z")
    plt.ylabel(r"$\mu$")
    plt.legend()

    if save:
        plt.savefig(
            GRAPHS_DIR / f"hubble_diagram.png",
            bbox_inches="tight",
        )
        plt.cla()


def velocity_vs_distance_graph(data, correction=Correction.ONE, save=True):
    c = 299792.458  # km/s
    plt.rcParams["figure.figsize"] = (8, 6)

    #xs = [sn.mu_distance(Correction.TWO) for sn in data]
    #ys = [sn.energy_loss() / sn.inflation_rate() for sn in data]

    xs = [sn.z for sn in data]
    ys = [sn.mu_distance(Correction.ONE) for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    dist_per_z = coefficients.slope

    xs = [sn.mu_distance() for sn in data]
    ys = [sn.delta_distance() for sn in data]

    #xs = [sn.energy_loss() for sn in data]
    #ys = [6.67146866 * sn.mu_distance(Correction.ONE) * sn.inflation_rate() for sn in data]

    #xs = [sn.mu_distance(Correction.NONE) for sn in data]
    #ys = [sn.inflation_rate() + 0.00004035447 * sn.energy_loss() for sn in data]

    #xs = [sn.mu_distance(Correction.NONE) for sn in data]
    #ys = [sn.energy_loss() / (6.428291819047782 * sn.mu_distance(Correction.ONE)) - sn.inflation_rate() for sn in data]

    #xs = [sn.mu_distance(Correction.NONE) for sn in data]
    #ys = [sn.inflation_rate() - 3.7806029237107605e-05 / (1 + sn.z) for sn in data]

    #xs = [sn.mu_distance(Correction.NONE) for sn in data]
    #ys = [sn.inflation_rate() - 3.7806029237107605e-05 / sn.mu_distance(Correction.ONE) for sn in data]


    xmin = min(xs)
    xmax = max(xs)

    plt.scatter(
        xs,
        ys,
        s=15,
        marker="^",
        linewidths=0.7,
        facecolors="none",
        edgecolors="blue",
        label="k(z) = 1",
    )

    p = np.polyfit(xs, ys, 2)
    xfit = np.linspace(xmin, xmax, 1000)
    yfit = np.polyval(p, xfit)
    plt.plot(xfit, yfit)

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    print(coefficients)
    plt.axline(
        (xmin, xmin * coefficients.slope + coefficients.intercept),
        (xmax, xmax * coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    plt.title("Linear Hubble constant")
    plt.xlabel("distance (Mpc)")
    plt.ylabel("velocity (km/s)")

    if save:
        plt.savefig(
            GRAPHS_DIR / f"velocity_vs_distance.png",
            bbox_inches="tight",
        )
        plt.cla()


def delta_distance_graph(data, save=True):
    # c = 299792.458  # km/s

    def delta(d_one, d_none):
        t = d_none / 9.71561e-9
        return math.log(d_none / d_one) / t

    plt.rcParams["figure.figsize"] = (8, 6)

    xs = [sn.mu_distance(correction=Correction.ONE) for sn in data]
    ys = [
        delta(sn.mu_distance(Correction.ONE), sn.mu_distance(Correction.NONE))
        for sn in data
    ]

    plt.scatter(
        xs,
        ys,
        s=15,
        marker="^",
        linewidths=0.7,
        facecolors="none",
        edgecolors="blue",
        label="k(z) = 1",
    )


def generate_all_graphs(data):
    all_mu_distance_vs_redshift_graph(data, save=True)
    velocity_vs_distance_graph(data, save=True)
    # hubble_diagram_graph(data, save=save)

def my_model(data, save=True):
    xs = [sn.z for sn in data]
    ys = [sn.mu_distance() for sn in data]

    xs = [math.log(x) for x in xs]
    ys = [math.log(y) for y in ys]

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    b, m = coefficients.intercept, coefficients.slope
    ys = [y - m * x - b for x, y in zip(xs, ys)]

    q1 = np.percentile(ys, 25)
    q3 = np.percentile(ys, 75)
    iqr = q3 - q1

    data = [sn for sn, y in zip(data, ys) if q1 - 1.5 * iqr <= y <= q3 + 1.5 * iqr]

    xs = [sn.z for sn in data]
    ys = [sn.mu_distance() for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    b, m = coefficients.intercept, coefficients.slope

    xs = [math.log(x) for x in xs]
    ys = [math.log(y) for y in ys]

    m, _ = np.polyfit(xs, ys, 1)
    ys = [y - m * x for x, y in zip(xs, ys)]

    _, sigma = scipy.stats.norm.fit(ys)
    xs = [sn.z for sn in data]
    ys = [sn.mu_distance() for sn in data]
    m, b = scipy.stats.siegelslopes(x=xs, y=ys)
    return data, m, b, sigma

def graph_model(data):
    xs = [sn.z for sn in data]
    ys = [sn.mu_distance() for sn in data]

    xmin, xmax = min(xs), max(xs)
    plt.scatter(
        xs,
        ys,
        s=15,
        marker="^",
        linewidths=0.7,
        facecolors="none",
        edgecolors="blue",
        label="k(z) = 1",
    )

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    plt.axline(
        (xmin, xmin * coefficients.slope + coefficients.intercept),
        (xmax, xmax * coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    data, m, b, sigma = my_model(data)
    xs = [sn.z for sn in data]
    ys = [m * x * np.e ** (scipy.stats.norm(0, sigma).rvs()) + b for x in xs]
    plt.scatter(xs, ys, s=15, marker="x", linewidths=0.7, color="red")


def graph(data):
    xs = [sn.z for sn in data]
    ys = [sn.x1 for sn in data]

    xmin, xmax = min(xs), max(xs)
    plt.scatter(
        xs,
        ys,
        s=15,
        marker="^",
        linewidths=0.7,
        facecolors="none",
        edgecolors="blue",
        label="k(z) = 1",
    )

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    print(coefficients)
    plt.axline(
        (xmin, xmin * coefficients.slope + coefficients.intercept),
        (xmax, xmax * coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )


def energy_loss_vs_orig_distance(data, correction=Correction.ONE, save=True):
    c = 299792.458  # km/s
    plt.rcParams["figure.figsize"] = (8, 6)

    xs = [sn.orig_distance() for sn in data]
    ys = [sn.energy_loss() for sn in data]

    xmin = min(xs)
    xmax = max(xs)

    plt.scatter(
        xs,
        ys,
        s=15,
        marker="^",
        linewidths=0.7,
        facecolors="none",
        edgecolors="blue",
    )

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    print(coefficients)
    print(f"If linear, 100% energy loss occurs for distances greater than {1 / coefficients.slope} parsecs.")
    plt.axline(
        (xmin, xmin * coefficients.slope + coefficients.intercept),
        (xmax, xmax * coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    plt.title("Linear Hubble constant")
    plt.xlabel("Original distance (Mpc)")
    plt.ylabel("energy loss")

    if save:
        plt.savefig(
            GRAPHS_DIR / f"energy_loss_vs_orig_distance.png",
            bbox_inches="tight",
        )
        plt.cla()


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (80, 60)
    plt.rcParams["text.usetex"] = True

    data = Supernova.from_abbott()

    #data, m, b, sigma = my_model(data)

    #generate_all_graphs(data)

    #all_mu_distance_vs_redshift_graph(data, save=False)
    velocity_vs_distance_graph(data, correction=Correction.ONE, save=False)
    #energy_loss_vs_orig_distance(data, correction=Correction.ONE, save=False)

    #graph_model(data)

    #graph(data)

    plt.show()
