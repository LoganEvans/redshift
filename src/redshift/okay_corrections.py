from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pprint import pprint
from redshift import supernova
import bisect
import csv
import jdcal
import numpy as np
import os
import pathlib
import scipy
import snpy
import math


REPO_DIR = pathlib.Path(os.environ["VIRTUAL_ENV"]).parent.resolve()
DATA_DIR = REPO_DIR / "data"
FILTERS_DIR = DATA_DIR / "filters" / "suprime-cam"


# 2001fo,0.772,I,-3.4,20.419540229885058,21.72488510198318
mc_name = "2001fo"
mc_epoch = -3.4
mc_z = 0.772
mc_rest_frame_counts = {"B": 238695, "I": 48180, "R": 88367, "V": 128791, "z": 39860}
mc_obs_frame_counts = {"B": 12967, "I": 94417, "R": 100569, "V": 66972, "z": 63340}


def parse_jd_date(jd):
    if isinstance(jd, str):
        jd = float(jd)
    year, month, day, _ = jdcal.jd2gcal(jd, 0)
    return date(year=year, month=month, day=day)


@dataclass()
class Filter:
    name: str
    wavelengths: list[float] = field(init=False)
    sensitivities: list[float] = field(init=False)

    def __post_init__(self):
        fpath = FILTERS_DIR / f"{self.name}.txt"
        self.wavelengths = []
        self.sensitivities = []
        with open(fpath, "r") as fin:
            for line in fin.readlines():
                vals = line.split()
                self.wavelengths.append(float(vals[0]))
                self.sensitivities.append(float(vals[1]))

    def __repr__(self):
        return f"Filter({self.name})"

    def sensitivity(self, wavelength):
        idx = bisect.bisect_right(self.wavelengths, wavelength) - 1
        return self.sensitivities[idx]

    _get_cache = {}

    @staticmethod
    def get(name):
        if name == "Z":
            name = "z"
        return Filter._get_cache.get(name, Filter(name))


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
                        name=row[idx["SN"]],
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
    epoch: float
    kcorr_filt: Filter | None = field(default=None)
    kcorr_mag: float | None = field(default=None)
    kcorr_mag_std_dev: float | None = field(default=None)

    @staticmethod
    def from_file():
        fpath = DATA_DIR / "barris-2004" / "table8.dat"

        idx = {
            "name": 1,
            "jd": 2,
            "filter": 3,
            "flux": 4,
            "flux_std_dev": 5,
            "epoch": 6,
            "kcorr_filter": 7,
            "kcorr_mag": 8,
            "kcorr_mag_std_dev": 9,
        }

        observations = []
        with open(fpath, "r") as fin:
            for line in fin.readlines():
                vals = line.split()
                if not vals:
                    continue

                args = {
                    "name": vals[idx["name"]],
                    "timestamp": parse_jd_date(2400000 + float(vals[idx["jd"]])),
                    "filt": Filter.get(vals[idx["filter"]]),
                    "flux": float(vals[idx["flux"]]),
                    "flux_std_dev": float(vals[idx["flux_std_dev"]]),
                    "epoch": float(vals[idx["epoch"]]),
                }

                if len(vals) == 10:
                    args["kcorr_filt"] = Filter.get(vals[idx["kcorr_filter"]])
                    args["kcorr_mag"] = float(vals[idx["kcorr_mag"]])
                    args["kcorr_mag_std_dev"] = float(vals[idx["kcorr_mag_std_dev"]])

                observations.append(Observation(**args))

        return observations


# table6.dat
_zs = {
    "2001fo": 0.772,
    "2001fs": 0.874,
    "2001hs": 0.833,
    "2001hu": 0.882,
    "2001hx": 0.799,
    "2001hy": 0.812,
    "2001iv": 0.3965,
    "2001iw": 0.3396,
    "2001ix": 0.711,
    "2001iy": 0.568,
    "2001jb": 0.698,
    "2001jf": 0.815,
    "2001jh": 0.885,
    "2001jm": 0.978,
    "2001jn": 0.645,
    "2001jp": 0.528,
    "2001kd": 0.936,
    "2002P": 0.719,
    "2002W": 1.031,
    "2002X": 0.859,
    "2002aa": 0.946,
    "2002ab": 0.423,
    "2002ad": 0.514,
}


@dataclass()
class Supernova:
    name: str
    observations: list[Observation]

    def __post_init__(self):
        for obs in self.observations:
            assert self.name == obs.name

    @staticmethod
    def get_all():
        supernovas = []

        observations = Observation.from_file()
        data = {}
        for obs in observations:
            if obs.name not in data:
                data[obs.name] = []
            data[obs.name].append(obs)

        for name, sn_observations in data.items():
            ten_day_obs = []
            for obs in sn_observations:
                if abs(obs.epoch) < 10:
                    ten_day_obs.append(obs)

            supernovas.append(Supernova(name=name, observations=ten_day_obs))

        return supernovas

    @property
    def z(self):
        return _zs[self.name]


def frame_counts_deterministic(z, epoch, wave=None, flux=None):
    if wave is None:
        wave, flux = snpy.getSED(epoch, "H3")

    filt_names = ["B", "I", "R", "V", "z"]
    rest_frame_counts = {filt_name: 0 for filt_name in filt_names}
    obs_frame_counts = {filt_name: 0 for filt_name in filt_names}

    for filt_name in filt_names:
        filt = Filter.get(filt_name)

        wls = sorted(set(wave).union(set(filt.wavelengths)))
        flxs = [0.0 for _ in wls]

        wls_idx = 0
        for flux_idx in range(len(flux)):
            wls_idxs = []

            if flux_idx + 1 == len(flux):
                wls_idxs = list(range(wls_idx, len(wls) - 1))
            else:
                while wls_idx + 1 < len(wls) and wls[wls_idx + 1] <= wave[flux_idx + 1]:
                    wls_idxs.append(wls_idx)
                    wls_idx += 1

            flux_per_bucket = flux[flux_idx] / len(wls_idxs)
            for wls_idx in wls_idxs:
                flxs[wls_idx] += flux_per_bucket

        wls_idx = 0
        for filt_idx in range(len(filt.wavelengths) - 1):
            sensitivity = filt.sensitivities[filt_idx]

            while wls_idx + 1 < len(wls) and wls[wls_idx] < filt.wavelengths[filt_idx + 1]:
                rest_frame_counts[filt_name] += sensitivity * flxs[wls_idx]
                wls_idx += 1

    return rest_frame_counts, obs_frame_counts


_frame_counts_cache = {}


def frame_counts_monte_carlo(z, epoch, trials, time_dilation):
    cache_key = (z, epoch, trials, time_dilation)
    if cache_key in _frame_counts_cache:
        return _frame_counts_cache[cache_key]

    wave, flux = snpy.getSED(epoch, "H3")
    if flux is None:
        return None

    flux_cdf = [0.0]
    total_flux = 0.0
    for i, f in enumerate(flux):
        total_flux += f
        flux_cdf.append(total_flux)
    flux_cdf_dist = scipy.stats.uniform(0, total_flux)
    unif_dist = scipy.stats.uniform(0, 1)

    def get_idx(cdf):
        return bisect.bisect_right(flux_cdf, cdf) - 1

    # Make sure get_idx is correct.
    for i in range(len(flux_cdf) - 1):
        left = flux_cdf[i]
        right = flux_cdf[i + 1]
        assert i == get_idx(left)
        assert i == get_idx((left + right) / 2.0)

    start = datetime(year=1930, month=6, day=14)

    def gen_photon():
        flux_cdf_rv = flux_cdf_dist.rvs()

        idx = get_idx(flux_cdf_rv)
        assert flux_cdf[idx] <= flux_cdf_rv < flux_cdf[idx + 1]

        remaining = flux_cdf_rv - flux_cdf[idx]
        assert remaining <= flux[idx]

        # frac should be a uniform(0, 1) variable.
        frac = remaining / flux[idx]
        assert 0.0 <= frac <= 1.0

        bucket_width = wave[1] - wave[0]
        wavelength = wave[idx] + frac * bucket_width
        timestamp = start + timedelta(minutes=unif_dist.rvs())
        return (wavelength, timestamp)

    end = datetime(year=2024, month=11, day=2)

    def redshift(photon, z, time_dilation):
        wavelength = photon[0] * (1 + z)
        if time_dilation:
            timestamp = end + (photon[1] - start) * (1 + z)
        else:
            timestamp = end + (photon[1] - start)
        return (wavelength, timestamp)

    filt_names = ["B", "I", "R", "V", "z"]
    rest_frame_counts = {name: 0 for name in filt_names}
    obs_frame_counts = {name: 0 for name in filt_names}

    for _ in range(trials):
        photon = gen_photon()
        red_photon = redshift(photon, z=z, time_dilation=True)

        for filt_name in filt_names:
            filt = Filter.get(filt_name)

            if unif_dist.rvs() < filt.sensitivity(photon[0]):
                rest_frame_counts[filt_name] += 1

            if (red_photon[1] - end) <= timedelta(
                minutes=1
            ) and unif_dist.rvs() < filt.sensitivity(red_photon[0]):
                obs_frame_counts[filt_name] += 1

    result = rest_frame_counts, obs_frame_counts
    _frame_counts_cache[cache_key] = result
    return result


def flux_to_magnitude(flux):
    try:
        return -2.5 * math.log(flux, 10) + 25.0
    except ValueError:
        print(f"Error on flux: {flux}")
        raise


if __name__ == "__main__":
    print(f"name,z,filt,epoch,flux,mag")
    for sn in Supernova.get_all():
        for obs in sn.observations:
            if obs.flux <= 0.0:
                continue

            # counts = frame_counts_monte_carlo(
            #    z=sn.z, epoch=obs.epoch, trials=1000, time_dilation=True
            # )
            counts = frame_counts_deterministic(z=sn.z, epoch=obs.epoch)
            rf = counts[0]
            mult = mc_rest_frame_counts["B"] / rf["B"]
            rf = {key: int(count * mult) for key, count in rf.items()}

            print(mc_rest_frame_counts)
            print(rf)
            exit()

            if not counts:
                continue

            rest_frame_counts, obs_frame_counts = counts
            if obs_frame_counts[obs.filt.name] == 0:
                continue
            pprint(rest_frame_counts)
            pprint(obs_frame_counts)

            flux = obs.flux * rest_frame_counts["B"] / obs_frame_counts[obs.filt.name]

            print(
                f"{sn.name},{sn.z},{obs.filt.name},{obs.epoch},{flux},{flux_to_magnitude(flux)}"
            )
            exit()
