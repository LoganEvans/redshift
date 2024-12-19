from __future__ import annotations
from bisect import bisect
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import date
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from pprint import pprint
from typing import ClassVar
from enum import Enum
import csv
import json
import math
import numpy as np
import os
import pathlib
import random
import scipy

REPO_DIR = pathlib.Path(os.environ["VIRTUAL_ENV"]).parent.resolve()
DATA_DIR = REPO_DIR / "data"
GRAPHS_DIR = REPO_DIR / "paper"


class Correction(Enum):
    NONE = 0
    ONE = 1  # Single correction factor: 1+z
    TWO = 2  # Two correction factors: (1+z)^2


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

    @property
    def velocity_kms(self):
        c = 299792.458  # km/s

        z2 = self.z * self.z

        return (c * z2 + 2 * c * self.z) / (z2 + 2 * self.z + 2)

    def mu(self, correction=Correction.NONE):
        return self.corrected_magnitude(correction) - self.absolute_magnitude

    def lum_distance(self, correction=Correction.ONE, use_mpc=True):
        d = 10 ** (1 + self.mu(correction) / 5)
        if use_mpc:
            d /= 1e6
        return d

    def orig_distance(self, correction=Correction.ONE):
        return self.lum_distance(correction) / math.exp(self.z)

    def comoving_distance(self, correction=Correction.ONE):
        return self.orig_distance() * (1 + self.z)

    def delta_distance(self, correction=Correction.ONE):
        return self.lum_distance(correction) - self.orig_distance(correction)

    def energy_loss(self):
        # E = hc / l
        # E_loss = hc/l - hc/(l*(1+z))
        # E_loss = (hc(1+z) - hc) / (l(1+z))
        # E_loss = hcz / (l(1+z))
        # E_loss = hc/l * (z / (1+z))
        return self.z / (1 + self.z)

    @property
    def t(self):
        return self.lum_distance(correction=Correction.ONE, use_mpc=False) * 3.26156

    @property
    def rate(self):
        return self.z / self.t

    @property
    def hubble(self):
        return (
            self.rate
            * self.lum_distance(use_mpc=False)
            * 3.086e13
            / (365 * 24 * 60 * 60)
        )

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
                    # see perivolaropoulos2022 for the Camerena and Marra references
                    # absolute_magnitude=-19.105, # ~70km/s / Mpc
                    absolute_magnitude=-19.2334, # Camarena and Marra 2020b
                    # absolute_magnitude=-19.3232,  # Abbott
                    # absolute_magnitude=-19.401, # Camarena and Marra 2020a
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


def k_corrections_for_photon_counts_graph(save=False):
    lo = 100
    hi = 2000
    xs = np.linspace(lo, hi, hi - lo)

    pdf = scipy.stats.norm(500, 100)
    fig, axs = plt.subplots(2, 2)

    fig.suptitle("k-corrections for photon counts")

    z = 1

    p_per = 100000

    ps_00 = np.zeros(len(xs))
    for i, nm in enumerate(xs):
        nm_upper = (nm + 1) / 2
        nm_lower = nm / 2
        cdf_upper = pdf.cdf(nm_upper)
        cdf_lower = pdf.cdf(nm_lower)
        ps_00[i] = p_per * (cdf_upper - cdf_lower) / 2

    axs[0, 0].sharex(axs[1, 1])
    axs[0, 0].sharey(axs[1, 1])
    axs[0, 0].plot(xs, ps_00, "tab:green")
    axs[0, 0].set_title(r"Observed photon counts per $\lambda$")
    axs[0, 0].set_xlabel(r"$\lambda$ (nm)")
    axs[0, 0].set_ylabel("photons")
    photon_count = sum(
        [photons if 1000 <= x <= 1200 else 0 for x, photons in zip(xs, ps_00)]
    )
    axs[0, 0].fill_between(
        xs,
        ps_00,
        where=[True if 1000 <= x <= 1200 else False for x in xs],
        facecolor="darkgreen",
        edgecolor="lightgreen",
        hatch=r"//",
        label=f"photons/nm = {photon_count:.0f}/200nm = {photon_count // 200:.0f}/nm",
    )
    axs[0, 0].legend()

    ps_10 = np.zeros(len(xs))
    for i, nm in enumerate(xs):
        nm_upper = (nm + 1) / 2
        nm_lower = nm / 2
        cdf_upper = pdf.cdf(nm_upper)
        cdf_lower = pdf.cdf(nm_lower)
        ps_10[i] = p_per * (cdf_upper - cdf_lower)

    axs[1, 0].sharex(axs[1, 1])
    axs[1, 0].sharey(axs[1, 1])
    axs[1, 0].plot(xs, ps_10, "tab:red")
    axs[1, 0].set_title("Corrected for time dilation")
    axs[1, 0].set_xlabel(r"$\lambda$ (nm)")
    axs[1, 0].set_ylabel("photons")
    photon_count = sum(
        [photons if 1000 <= x <= 1200 else 0 for x, photons in zip(xs, ps_10)]
    )
    axs[1, 0].fill_between(
        xs,
        ps_10,
        where=[True if 1000 <= x <= 1200 else False for x in xs],
        facecolor="darkred",
        edgecolor="red",
        hatch=r"//",
        label=f"photons/nm = {photon_count:.0f}/200nm = {photon_count // 200:.0f}/nm",
    )
    axs[1, 0].legend()

    ps_01 = np.zeros(len(xs))
    for i, nm in enumerate(xs):
        nm_upper = nm + 1
        nm_lower = nm
        cdf_upper = pdf.cdf(nm_upper)
        cdf_lower = pdf.cdf(nm_lower)
        ps_01[i] = p_per * (cdf_upper - cdf_lower) / 2

    axs[0, 1].sharex(axs[1, 1])
    axs[0, 1].sharey(axs[1, 1])
    axs[0, 1].plot(xs, ps_01, "tab:red")
    axs[0, 1].set_title("Corrected for redshift")
    axs[0, 1].set_xlabel(r"$\lambda$ (nm)")
    axs[0, 1].set_ylabel("photons")
    photon_count = sum(
        [photons if 500 <= x <= 600 else 0 for x, photons in zip(xs, ps_01)]
    )
    axs[0, 1].fill_between(
        xs,
        ps_01,
        where=[True if 500 <= x <= 600 else False for x in xs],
        facecolor="darkred",
        edgecolor="red",
        hatch=r"//",
        label=f"photons/nm = {photon_count:.0f}/100nm = {photon_count // 100:.0f}/nm",
    )
    axs[0, 1].legend()

    ps_11 = np.zeros(len(xs))
    for i, nm in enumerate(xs):
        nm_upper = nm + 1
        nm_lower = nm
        cdf_upper = pdf.cdf(nm_upper)
        cdf_lower = pdf.cdf(nm_lower)
        ps_11[i] = p_per * (cdf_upper - cdf_lower)

    axs[1, 1].plot(xs, ps_11, "tab:blue")
    axs[1, 1].set_title(r"Corrected for redshift and time dilation")
    axs[1, 1].set_xlabel(r"$\lambda$ (nm)")
    axs[1, 1].set_ylabel("photons")
    photon_count = sum(
        [photons if 500 <= x <= 600 else 0 for x, photons in zip(xs, ps_11)]
    )
    axs[1, 1].fill_between(
        xs,
        ps_11,
        where=[True if 500 <= x <= 600 else False for x in xs],
        facecolor="darkblue",
        edgecolor="lightblue",
        hatch=r"//",
        label=f"photons/nm = {photon_count:.0f}/100nm = {photon_count // 100:.0f}/nm",
    )
    axs[1, 1].legend()

    if save:
        plt.savefig(
            GRAPHS_DIR / f"k-corrections_for_photon_counts.png",
            bbox_inches="tight",
        )
        plt.cla()


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


def all_lum_distance_vs_redshift_graph(data, save=True):
    plt.rcParams["figure.figsize"] = (8, 6)

    reduced_data = reduce_data(data)
    xs = [sn.z for sn in reduced_data]

    ys = [sn.lum_distance(correction=Correction.ONE) for sn in reduced_data]
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
        y=[sn.lum_distance(correction=Correction.ONE) for sn in data],
    )
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    ys = [sn.lum_distance(correction=Correction.NONE) for sn in reduced_data]
    plt.scatter(xs, ys, s=15, marker="x", linewidths=0.7, color="red", label="k(z) = 1")
    coefficients = scipy.stats.siegelslopes(
        x=[sn.z for sn in data],
        y=[sn.lum_distance(correction=Correction.NONE) for sn in data],
    )
    print(coefficients.slope)
    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    # ys = [sn.lum_distance(correction=Correction.TWO) for sn in data]
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
    #    y=[sn.lum_distance(correction=Correction.TWO) for sn in old_data],
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
        plt.savefig(GRAPHS_DIR / "lum_distance_vs_redshift.png", bbox_inches="tight")
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


def recessional_velocity_vs_intercept_time_graph(
    data, correction=Correction.ONE, save=True
):
    plt.rcParams["figure.figsize"] = (8, 6)

    xs = [sn.velocity_kms for sn in data]
    ys = [sn.lum_distance(correction) / sn.velocity_kms for sn in data]

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    dist_per_z = coefficients.slope

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
        label="k(z) = 1+z",
    )

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    plt.axline(
        (xmin, xmin * coefficients.slope + coefficients.intercept),
        (xmax, xmax * coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
    )

    if save:
        plt.savefig(
            GRAPHS_DIR / f"recessional_velocity_vs_intercept_time.png",
            bbox_inches="tight",
        )
        plt.cla()


def velocity_vs_distance_graph(data, correction=Correction.ONE, save=True):
    c = 299792.458  # km/s
    plt.rcParams["figure.figsize"] = (8, 6)

    mean_rate = np.mean([sn.rate for sn in data])

    xs = [sn.lum_distance() for sn in data]
    ys = [sn.hubble for sn in data]

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    dist_per_z = coefficients.slope

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
        label="k(z) = 1+z",
    )

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    plt.axline(
        (xmin, xmin * coefficients.slope + coefficients.intercept),
        (xmax, xmax * coefficients.slope + coefficients.intercept),
        color="black",
        linewidth=0.5,
        linestyle="--",
        label=f"$v = {coefficients.slope:.2f} \\frac{{km/s}}{{Mpc}} \\times D_l + {coefficients.intercept:.0f} km/s$",
    )

    plt.title("Linear Hubble constant")
    plt.xlabel("distance (Mpc)")
    plt.ylabel("velocity (km/s)")
    plt.legend()

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

    xs = [sn.lum_distance(correction=Correction.ONE) for sn in data]
    ys = [
        delta(sn.lum_distance(Correction.ONE), sn.lum_distance(Correction.NONE))
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


def bootstrap_hubble_parameter_graph(data, save=False):
    trials = 100000

    # From https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.013028
    # Local determination of the Hubble constant and the deceleration parameter,
    # David Camarena and Valerio Marra
    abs_mag_dist = scipy.stats.norm(-19.2334, 0.0404)

    results = []
    for i in range(trials):
        print(f"Trial: {i:5d} / {trials}", end="\r")

        abs_mag = abs_mag_dist.rvs()
        new_data = [dataclasses.replace(sn) for sn in data]
        for sn in new_data:
            sn.absolute_magnitude = abs_mag
        sample = random.choices(new_data, k=len(data))
        coefficients = scipy.stats.siegelslopes(
            x=[sn.lum_distance() for sn in sample],
            y=[sn.hubble for sn in sample],
        )
        results.append(coefficients.slope)

    print("H0 normal distribution fit:")
    print(scipy.stats.norm.fit(results))

    plt.hist([results], bins=200)
    plt.title("Bootstrapped H0")
    plt.xlabel(r"$\frac{km/s}{Mpc}$")
    if save:
        plt.savefig(
            GRAPHS_DIR / f"bootstrapped_H0.png",
            bbox_inches="tight",
        )
        plt.cla()


def generate_all_graphs(data):
    figsize = (12, 9)
    linewidth = 4

    plt.cla()
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["text.usetex"] = True
    plt.rcParams["hatch.linewidth"] = linewidth
    all_lum_distance_vs_redshift_graph(data, save=True)

    plt.cla()
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["text.usetex"] = True
    plt.rcParams["hatch.linewidth"] = linewidth
    velocity_vs_distance_graph(data, save=True)

    #plt.cla()
    #plt.rcParams["figure.figsize"] = figsize
    #plt.rcParams["text.usetex"] = True
    #plt.rcParams["hatch.linewidth"] = linewidth
    #bootstrap_hubble_parameter_graph(data, save=True)

    plt.cla()
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["text.usetex"] = True
    plt.rcParams["hatch.linewidth"] = linewidth
    k_corrections_for_photon_counts_graph(save=True)


def my_model(data, save=True):
    xs = [sn.z for sn in data]
    ys = [sn.lum_distance() for sn in data]

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
    ys = [sn.lum_distance() for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    b, m = coefficients.intercept, coefficients.slope

    xs = [math.log(x) for x in xs]
    ys = [math.log(y) for y in ys]

    m, _ = np.polyfit(xs, ys, 1)
    ys = [y - m * x for x, y in zip(xs, ys)]

    _, sigma = scipy.stats.norm.fit(ys)
    xs = [sn.z for sn in data]
    ys = [sn.lum_distance() for sn in data]
    m, b = scipy.stats.siegelslopes(x=xs, y=ys)
    return data, m, b, sigma


def graph_model(data):
    xs = [sn.z for sn in data]
    ys = [sn.lum_distance() for sn in data]

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
    xs = [sn.comoving_distance() for sn in data]
    ys = [sn.lum_distance() for sn in data]

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
    print(
        f"If linear, 100% energy loss occurs for distances greater than {1 / coefficients.slope} parsecs."
    )
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
    plt.rcParams["figure.figsize"] = (16, 12)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["hatch.linewidth"] = 4

    data = Supernova.from_abbott()

    #generate_all_graphs(data)

    #all_lum_distance_vs_redshift_graph(data, save=False)
    # velocity_vs_distance_graph(data, correction=Correction.ONE, save=False)
    # energy_loss_vs_orig_distance(data, correction=Correction.ONE, save=False)
    # recessional_velocity_vs_intercept_time_graph(data, correction=Correction.ONE, save=False)

    # graph_model(data)

    graph(data)

    #bootstrap_hubble_parameter_graph(data)
    plt.show()
