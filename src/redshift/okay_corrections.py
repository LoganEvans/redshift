from dataclasses import dataclass, field
import csv
from datetime import date, date, timedelta
from pprint import pprint
import jdcal
from redshift import supernova
import pathlib
import os


REPO_DIR = pathlib.Path(os.environ["VIRTUAL_ENV"]).parent.resolve()
DATA_DIR = REPO_DIR / "data"
LC_DIR = DATA_DIR / "cfalc_allsnIa"
SPEC_DIR = DATA_DIR / "cfaspec_M08snIa"


def parse_jd_date(jd: str):
    year, month, day, _ = jdcal.jd2gcal(float(jd), 0)
    return date(year=year, month=month, day=day)


@dataclass()
class LightCurveBand:
    band: str
    mag: float
    std_dev: float


@dataclass()
class LightCurve:
    timestamp: date
    bands: dict[str, LightCurveBand]


@dataclass()
class Spec:
    timestamp: date
    wave: list[float]
    spec: list[float]


@dataclass()
class SnInfo:
    name: str
    sn_type: str
    host: str
    discovery_timestamp: date
    max_mag_timestamp: date
    max_mag: float
    last_mag_timestamp: date
    last_mag: float
    z: float
    z_host: float
    reference: str

    @staticmethod
    def all_info():
        if not hasattr(SnInfo, "_all_info_cache"):
            SnInfo._all_info_cache = {}

            with open(DATA_DIR / "rochester-data.csv", "r") as csv_file:
                reader = csv.reader(csv_file)

                header = next(reader)
                idx = {field: i for i, field in enumerate(header)}

                for row in reader:
                    sn_info = SnInfo(
                        name=row[idx["SN"]].upper(),
                        sn_type=row[idx["Type"]],
                        host=row[idx["Host"]],
                        discovery_timestamp=parse_jd_date(
                            row[idx["Discovery date (JD)"]]
                        ),
                        max_mag_timestamp=parse_jd_date(row[idx["Max mag date (JD)"]]),
                        max_mag=float(row[idx["Max mag"]]),
                        last_mag_timestamp=parse_jd_date(
                            row[idx["Last mag date (JD)"]]
                        ),
                        last_mag=float(row[idx["Last mag"]]),
                        z=None if row[idx["z"]] == "n/a" else float(row[idx["z"]]),
                        z_host=(
                            None
                            if row[idx["z host"]] == "n/a"
                            else float(row[idx["z host"]])
                        ),
                        reference=row[idx["Reference"]],
                    )
                    SnInfo._all_info_cache[sn_info.name] = sn_info

        return SnInfo._all_info_cache


@dataclass()
class Observation:
    name: str
    timestamp: date
    filt: Filter
    flux: float
    flux_std_dev: float
    epoch: timedelta
    kcorr_filt: Filter | None = field(default=None)
    kcorr_mag: float | None = field(default=None)
    kcorr_mag_std_dev: float | None = field(default=None)


@dataclass()
class Supernova:
    name: str
    sn_info: SnInfo = field(init=False)
    observations: list[Observation] = field(init=False)

    def __post_init__(self):
        self.sn_info = SnInfo.all_info()[self.name.upper().lstrip("SN")]

        light_curves = {}

        for fname in LC_DIR.iterdir():
            if fname.stem.startswith(self.name):
                bands = fname.stem.split("_")[-1]
                header = ["ja"]
                for band in bands:
                    header.append(band)
                    header.append(f"{band}_std_dev")

                with open(fname, "r") as fin:
                    use_hjd = None
                    use_2400000 = None
                    use_2450000 = None

                    for line in fin.readlines():
                        if "HJD" in line:
                            use_hjd = True
                        elif "Julian Day" in line:
                            use_hjd = True
                        elif "JD-2400000" in line:
                            use_2400000 = True
                        elif "JD-2450000" in line:
                            use_2450000 = True

                        if line.startswith("#"):
                            continue
                        vals = line.split()

                        data = {}
                        for col, raw in zip(header, vals):
                            if col == "ja":
                                val = float(raw)
                                if use_2400000:
                                    val += 2400000
                                elif use_2450000:
                                    val += 2450000
                                data["timestamp"] = parse_jd_date(str(val))
                            else:
                                data[col] = float(raw)

                        bands_data = {}
                        for band in bands:
                            bands_data[band] = LightCurveBand(
                                band=band,
                                mag=data[band],
                                std_dev=data[f"{band}_std_dev"],
                            )

                        lc = LightCurve(timestamp=data["timestamp"], bands=bands_data)
                        light_curves[lc.timestamp] = lc
                break

        self.observations = []
        pass


if __name__ == "__main__":
    sns = []
    for f in SPEC_DIR.iterdir():
        if f.is_dir():
            sns.append(Supernova(name=f.stem))
    zs = [sn.sn_info.z for sn in sns if sn.sn_info.z]
    print(sorted(zs))
