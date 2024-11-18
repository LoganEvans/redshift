from __future__ import annotations
from dataclasses import dataclass
from matplotlib import pyplot as plt
import os
import math
from pprint import pprint
import csv
import numpy as np
import pathlib
import scipy
from enum import Enum

REPO_DIR = pathlib.Path(os.environ["VIRTUAL_ENV"]).parent.resolve()
DATA_DIR = REPO_DIR / "data"
GRAPHS_DIR = REPO_DIR / "graphs"


class Correction(Enum):
    NONE = 0
    RV = 1  # Recessional velocity
    TL = 2  # Tired light


@dataclass()
class Cepheid:
    field: str
    alpha: float
    delta: float
    id: float
    period: float
    V_minus_I: float
    F160W: float
    sigma: float
    offset: float
    bias: float
    IMrms: float
    o_h: float
    flag: str

    _field2z = {
        "n1309": 0.007125,
        "n3021": 0.00514,
        "n3370": 0.004266,
        "n3982": 0.003699,
        "n4038": 0.005477,
        "n4258": 0.001494,
        "n4536": 0.006031,
        "n4639": 0.003395,
        "n5584": 0.005464,
    }

    @staticmethod
    def from_riess():
        data = []
        with open(DATA_DIR / "riess_2011_cepheids.csv", "r") as fin:
            reader = csv.reader(fin)
            header = next(reader)
            idx = {val: i for i, val in enumerate(header)}

            for row in reader:
                params = {}
                for key in [
                    "alpha",
                    "delta",
                    "id",
                    "period",
                    "V_minus_I",
                    "F160W",
                    "sigma",
                    "offset",
                    "bias",
                    "IMrms",
                    "o_h",
                ]:
                    params[key] = float(row[idx[key]])

                for key in ["field", "flag"]:
                    params[key] = row[idx[key]]
                data.append(Cepheid(**params))
        return data

    def magnitude(self, correction=Correction.NONE):
        if correction == Correction.NONE:
            return self.F160W
        elif correction == Correction.RV:
            return self.F160W - 2 * np.log(1 + self.z) / np.log(100**0.2)
        else: # Tired light
            return self.F160W - np.log(1 + self.z) / np.log(100**0.2)

    @property
    def mean_magnitude(self):
        return -2.43 * (math.log(self.period, 10) - 1) - 4.05

    def distance(self, correction=Correction.NONE, majaess=True):
        if majaess:
            d = 10 ** (
                self.magnitude(correction)
                + 3.37 * math.log(self.period, 10)
                - 2.55 * self.V_minus_I
                + 10.48
            )
        else:
            d = 10 ** (
                self.magnitude(correction)
                + 3.34 * math.log(self.period, 10)
                - 2.45 * self.V_minus_I
                + 10.52
            )

        return d

    @property
    def z(self):
        return self._field2z[self.field]

    @staticmethod
    def _uncalibrated_distance(magnitude, z, correction=Correction.NONE):
        d = 10 ** (magnitude / 5)
        if correction == Correction.RV:
            d /= z + 1
        elif correction == Correction.TL:
            d /= np.sqrt(z + 1)
        return d

    def uncalibrated_distance(self, correction=Correction.NONE):
        return self._uncalibrated_distance(self.magnitude, self.z, correction)


if __name__ == "__main__":
    data = [d for d in Cepheid.from_riess() if d.flag != "rej"]

    #for field in Cepheid._field2z.keys():
    #    print(field)
    #    cs = [c for c in data if c.field == field]
    #    if not data:
    #        continue
    #    ys = sorted([c.distance for c in cs])
    #    ys = ys[:-len(ys) // 10]
    #    print(len(ys))

    #    min_ys = min(ys)
    #    delta_ys = max(ys) - min_ys
    #    ys = [(y - min_ys) / delta_ys for y in ys]

    #    bins = np.linspace(
    #        min(ys),
    #        max(ys),
    #        10,
    #    )

    #    plt.hist(
    #        ys,
    #        bins,
    #        alpha=0.2
    #    )

    field2c = {field: [] for field in Cepheid._field2z.keys()}
    for c in data:
        field2c[c.field].append(c)

    vals = []
    for field, cs in field2c.items():
        v = sorted(cs, key=lambda c: c.distance())
        vals.extend(v[len(v)//10:2*len(v)//10])

    xs = [c.z for c in vals]
    ys = [
        c.distance(correction=Correction.NONE)
        for c in vals
    ]
    plt.scatter(
           xs,
           ys,
           s=15,
           marker="x",
           linewidths=0.7,
           color="red",
       )
    ys = [
        c.distance(correction=Correction.TL)
        for c in vals
    ]
    plt.scatter(
           xs,
           ys,
           s=15,
           marker="^",
           linewidths=0.7,
           facecolors="none",
           edgecolors="blue",
       )
    bins = np.linspace(
        min(ys),
        max(ys),
        100,
    )

    #plt.hist(
    #    ys,
    #    bins,
    #)

    # m, b = np.polyfit(xs, ys, 1)
    # plt.axline(
    #    (min(xs), m * min(xs) + b),
    #    (max(xs), m * max(xs) + b),
    #    color="black",
    #    linewidth=0.5,
    # )

    plt.show()
