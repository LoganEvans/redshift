from dataclasses import dataclass, field
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
    redshift: float

    H0: ClassVar[float] = 70

    @property
    def z(self):
        return self.redshift

    def uncalibrated_distance(self, corrected):
        d = 10 ** (self.magnitude / 5)
        if corrected:
            d /= np.sqrt(self.z + 1)
        return d

    def calibrated_distance(self, slope, intercept, corrected):
        # XXX This is wrong.
        return (
            (self.uncalibrated_distance(corrected=corrected) - intercept)
            * self.H0
            / slope
        )

    @property
    def velocity_kms(self):
        c = 299792.458 # km/s
        return self.z * c

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
                    label = rule[1].strip("'")
                    if label in ["stretch", "color"]:
                        continue

                    if label == "supernova name":
                        fields["name"] = rule[2].strip("'")
                    else:
                        fields[rule[1].strip("'")] = rule[2]

                data.append(Supernova(**fields))

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


def uncalibrated_graph(data, corrected=True):
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
    plt.savefig(
        GRAPHS_DIR / f"{'corrected' if corrected else 'uncorrected'}_uncalibrated.png",
        bbox_inches="tight",
    )
    plt.cla()


def both_calibrated_redshift_vs_distance_graph(data):
    plt.cla()
    xs = [sn.redshift for sn in data]

    ys = [sn.uncalibrated_distance(corrected=False) for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    ys = [
        sn.calibrated_distance(
            coefficients.slope, coefficients.intercept, corrected=False
        )
        for sn in data
    ]
    plt.scatter(xs, ys, s=2, marker=".", color="red")

    ys = [sn.uncalibrated_distance(corrected=True) for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    ys = [
        sn.calibrated_distance(
            coefficients.slope, coefficients.intercept, corrected=True
        )
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
    plt.savefig(GRAPHS_DIR / "both_calibrated_distance_vs_redshift.png", bbox_inches="tight")
    plt.cla()


def both_calibrated_velocity_vs_distance_graph(data):
    plt.cla()
    xs = [sn.redshift for sn in data]
    velo = [sn.velocity_kms for sn in data]

    ys = [sn.uncalibrated_distance(corrected=False) for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    distance = [
        sn.calibrated_distance(
            coefficients.slope, coefficients.intercept, corrected=False
        )
        for sn in data
    ]
    plt.scatter(distance, velo, s=2, marker=".", color="red")

    ys = [sn.uncalibrated_distance(corrected=True) for sn in data]
    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    distance = [
        sn.calibrated_distance(
            coefficients.slope, coefficients.intercept, corrected=True
        )
        for sn in data
    ]
    plt.scatter(distance, velo, s=2, marker=".", color="blue")

    plt.xlabel("distance (Mpsc)")
    plt.ylabel("velocity (km/s)")

    plt.title("Velocity vs Distance")
    plt.legend(["uncorrected", "corrected", f"H0 = {Supernova.H0:.0f} km/s / Mpsc"])
    plt.savefig(GRAPHS_DIR / "both_calibrated_velocity_vs_distance.png", bbox_inches="tight")
    plt.cla()


def bootstrap_slope(data, corrected, trials):
    med = np.median([sn.z for sn in data])

    left = [sn for sn in data if sn.z <= med]
    right = [sn for sn in data if sn.z > med]

    trial_results = []
    for label, data in [(f"z <= {med:.2f}", left), (f"z > {med:.2f}", right)]:
        slopes = []
        for i in range(trials):
            print(f"Bootstraping trial ({label}, corrected={corrected}): {i}/{trials}", end="\r")
            sample = np.random.choice(data, size=len(data), replace=True)
            y = [sn.velocity_kms for sn in sample]
            coefficients = scipy.stats.siegelslopes(
                x=[sn.uncalibrated_distance(corrected=corrected) for sn in sample],
                y=y,
            )
            coefficients = scipy.stats.siegelslopes(
                x=[sn.calibrated_distance(slope=coefficients.slope, intercept=coefficients.intercept, corrected=corrected) for sn in sample],
                y=y
            )

            slopes.append(coefficients.slope)
        trial_results.append(slopes)
        print()

    return trial_results


def bootstrap_graphs(data):
    for corrected in [True, False]:
        trials = 10000
        med = np.median([sn.z for sn in data])
        left, right = bootstrap_slope(data, corrected, trials)

        bins = np.linspace(
            min(min(left), min(right)),
            max(max(left), max(right)),
            100,
        )

        plt.hist(
            left,
            bins,
            alpha=0.5,
            label=f"slopes z <= {med:.2f}\n(median: {np.median(left):.2f})",
        )
        plt.hist(
            right,
            bins,
            alpha=0.5,
            label=f"slopes z > {med:.2f}\n(median: {np.median(right):.2f})",
        )
        plt.legend()

        plt.title(f"Bootstrapped Velocity (km/s) per {'Corrected' if corrected else 'Uncorrected'} Distance (Mpsc)")
        plt.savefig(GRAPHS_DIR / f"bootstrapped_velocity_per_{'' if corrected else 'un'}corrected_distance.png", bbox_inches="tight")
        plt.cla()


def generate_all_graphs(data):
    uncalibrated_graph(data, corrected=True)
    uncalibrated_graph(data, corrected=False)
    both_calibrated_redshift_vs_distance_graph(data)
    both_calibrated_velocity_vs_distance_graph(data)
    #bootstrap_graphs(data)

if __name__ == "__main__":
    # data = Supernova.from_rochester()
    data = Supernova.from_wolfram()
    generate_all_graphs(data)

    #generate_distance_vs_redshift_graphs(data)
    #exit()

    exit()

    print("Samples:", len(data))

    xs = [sn.z for sn in data]
    ys = [sn.uncalibrated_distance(corrected=corrected) for sn in data]

    s = 100 / len(xs)

    plt.scatter(xs, ys, s=s, marker=".", color="black")

    # coefficients = np.polyfit(xs, ys, 1)
    # print(coefficients)

    # coefficients = scipy.stats.theilslopes(x=xs, y=ys)
    # print(coefficients)

    coefficients = scipy.stats.siegelslopes(x=xs, y=ys)
    print(coefficients)

    plt.axline(
        (0, coefficients.intercept),
        (1, coefficients.intercept + coefficients.slope),
        color="red",
        linewidth=0.5,
    )

    plt.show()
