import pathlib
from dataclasses import dataclass
from datetime import date
import csv
import jdcal
import numpy as np
import scipy

from redshift.scratch import Supernova as ScratchSupernova

scratch = ScratchSupernova.from_wolfram()


DATA_DIR = (pathlib.Path(__file__).parent.parent.parent / "data").resolve()


def parse_jd_date(jd: str):
    year, month, day, _ = jdcal.jd2gcal(float(jd), 0)
    return date(year=year, month=month, day=day)


@dataclass()
class Supernova:
    name: str
    type: str
    host: str | None
    location_ra: str
    location_decl: str
    offset: str
    discovery_date: date
    max_mag_date: date
    max_mag: float
    last_mag_date: date
    last_mag: float
    z: float | None
    z_host: float | None
    reference: str
    discoverers: str
    aka: str

    @property
    def is_suitable_Ia(self):
        return (
            self.type == "Ia"
            and None not in [self.z, self.max_mag]
            and (9.9 < self.max_mag < 99.9)
            and self.discoverers != "Supernova Legacy Project"
            and self.max_mag_date.year <= 2000
        )

    @staticmethod
    def from_csv():
        supernovas = []
        with open(DATA_DIR / "rochester-data.csv", "r") as csv_file:
            reader = csv.reader(csv_file)

            header = next(reader)
            idx = {field: i for i, field in enumerate(header)}

            for row in reader:
                z = row[idx["z"]]
                if z == "n/a":
                    z = None
                else:
                    z = float(z)

                z_host = row[idx["z host"]]
                if z_host == "n/a":
                    z_host = None
                else:
                    z_host = float(z_host)

                supernovas.append(
                    Supernova(
                        name=row[idx["SN"]],
                        type=row[idx["Type"]],
                        host=row[idx["Host"]],
                        location_ra=row[idx["R.A."]],
                        location_decl=row[idx["Decl."]],
                        offset=row[idx["Offset"]],
                        discovery_date=parse_jd_date(row[idx["Discovery date (JD)"]]),
                        max_mag_date=parse_jd_date(row[idx["Max mag date (JD)"]]),
                        max_mag=float(row[idx["Max mag"]]),
                        last_mag_date=parse_jd_date(row[idx["Last mag date (JD)"]]),
                        last_mag=float(row[idx["Last mag"]]),
                        z=z,
                        z_host=z_host,
                        reference=row[idx["Reference"]],
                        discoverers=row[idx["Discoverer(s)"]],
                        aka=row[idx["AKA"]],
                    )
                )
        return supernovas

    @property
    def corrected_uncalibrated_luminosity_distance(self):
        return (100 ** (self.max_mag / 10)) / np.sqrt((self.z + 1))


if __name__ == "__main__":
    from redshift.scratch import Supernova as ScratchSupernova

    scratch = ScratchSupernova.from_wolfram()

    supernovas = Supernova.from_csv()

    Ia = [sn for sn in supernovas if sn.is_suitable_Ia]
    count = {}
    for sn in Ia:
        count[sn.discoverers] = count.get(sn.discoverers, 0) + 1

    print([sn.z_host for sn in Ia])

    from pprint import pprint

    pprint(count)
    print("Samples:", len(Ia))

    from matplotlib import pyplot as plt

    xs = [sn.z for sn in Ia]
    ys = [sn.corrected_uncalibrated_luminosity_distance for sn in Ia]

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
