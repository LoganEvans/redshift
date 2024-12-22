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

FILT_NAMES = ["B", "I", "R", "V", "z"]

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
    wave_data: list[float] = field(init=False)
    resp_data: list[float] = field(init=False)

    def __post_init__(self):
        fpath = FILTERS_DIR / f"{self.name}.txt"
        self.wave_data = []
        self.resp_data = []
        with open(fpath, "r") as fin:
            for line in fin.readlines():
                vals = line.split()
                self.wave_data.append(float(vals[0]))
                self.resp_data.append(float(vals[1]))

    def __repr__(self):
        return f"Filter({self.name})"

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


def redistribute_flux(wave_data, fluxes, new_wavelengths):
    assert set(wave_data).issubset(set(new_wavelengths))

    delta = wave_data[-1] - wave_data[-2]
    wave_data = list(wave_data[:]) + [wave_data[-1] + delta, wave_data[-1] + 2 * delta]
    fluxes = list(fluxes[:]) + [0.0, 0.0]

    wls_idx = 0
    new_fluxes = [0.0 for _ in new_wavelengths]
    x = 0.0
    for flux_idx in range(len(fluxes) - 1):
        x += fluxes[flux_idx]
        flux_width = wave_data[flux_idx + 1] - wave_data[flux_idx]

        flux_in_bucket = fluxes[flux_idx]
        flux_seen = 0.0
        if wls_idx == len(new_wavelengths):
            new_fluxes[wls_idx - 1] += flux_in_bucket
            continue

        while wls_idx < len(new_wavelengths):
            if wls_idx + 1 == len(new_wavelengths):
                bucket_end_wl = wave_data[flux_idx + 1]
            else:
                bucket_end_wl = new_wavelengths[wls_idx + 1]
                if bucket_end_wl > wave_data[flux_idx + 1]:
                    break

            new_flux_width = bucket_end_wl - new_wavelengths[wls_idx]
            new_fluxes[wls_idx] += fluxes[flux_idx] * new_flux_width / flux_width
            flux_seen += fluxes[flux_idx] * new_flux_width / flux_width
            wls_idx += 1

    return new_fluxes


def frame_counts_deterministic(
    z,
    epoch=None,
    rest_photon_dens=None,
    obs_photon_dens=None,
    time_dilation=True,
    filters=None,
):
    if not filters:
        filters = {}
        for filt_name in FILT_NAMES:
            filters[filt_name] = Filter.get(filt_name)

    rest_frame_counts = {filt_name: 0 for filt_name in filters.keys()}
    obs_frame_counts = {filt_name: 0 for filt_name in filters.keys()}

    # TODO(lpe): Factor this loop out into a function and reuse below.
    for filt_name, filt in filters.items():
        wls_idx = 0
        for filt_idx in range(len(filt.wave_data) - 1):
            resp = filt.resp_data[filt_idx]

            while wls_idx < filt.wave_data[filt_idx + 1]:
                rest_frame_counts[filt_name] += resp * rest_photon_dens[wls_idx]
                wls_idx += 1

    for filt_name, filt in filters.items():
        wls_idx = 0
        for filt_idx in range(len(filt.wave_data) - 1):
            resp = filt.resp_data[filt_idx]

            while wls_idx < filt.wave_data[filt_idx + 1]:
                obs_frame_counts[filt_name] += resp * obs_photon_dens[wls_idx]
                wls_idx += 1

    return rest_frame_counts, obs_frame_counts


def frame_counts_monte_carlo(
    z, trials, rest_photon_dens=None, obs_photon_dens=None, filters=None
):
    cdf = [0.0]
    cumulative_dens = 0.0
    total_dens = sum(rest_photon_dens)
    for i, dens in enumerate(rest_photon_dens):
        cumulative_dens += dens
        cdf.append(cumulative_dens / total_dens)

    unif_dist = scipy.stats.uniform(0, 1)

    def get_idx(unif_rv):
        return bisect.bisect_right(cdf, unif_rv) - 1

    start = datetime(year=1930, month=6, day=14)

    def gen_photon():
        cdf_rv = unif_dist.rvs()

        idx = get_idx(cdf_rv)
        assert cdf[idx] <= cdf_rv < cdf[idx + 1]

        remaining = cdf_rv - cdf[idx]
        assert remaining <= rest_photon_dens[idx]

        # frac should be a uniform(0, 1) variable.
        frac = remaining / rest_photon_dens[idx]
        assert 0.0 <= frac <= 1.0

        wavelength = idx + frac
        timestamp = start + timedelta(minutes=unif_dist.rvs())
        return (wavelength, timestamp)

    end = datetime(year=2024, month=11, day=2)

    def redshift(photon, z):
        wavelength = photon[0] * (1 + z)
        timestamp = end + (photon[1] - start) * (1 + z)
        return (wavelength, timestamp)

    def sensitivity(filt, wavelength):
        idx = bisect.bisect_right(filt.wave_data, wavelength) - 1
        if idx == -1 or idx + 1 == len(filt.wave_data):
            return 0.0
        return filt.resp_data[idx]

    rest_frame_counts = {name: 0 for name in filters.keys()}
    obs_frame_counts = {name: 0 for name in filters.keys()}

    for _ in range(trials):
        photon = gen_photon()
        red_photon = redshift(photon, z=z)

        for filt_name, filt in filters.items():
            if unif_dist.rvs() < sensitivity(filt, photon[0]):
                rest_frame_counts[filt_name] += 1

            if (red_photon[1] - end) <= timedelta(
                minutes=1
            ) and unif_dist.rvs() < sensitivity(filt, red_photon[0]):
                obs_frame_counts[filt_name] += 1

    return rest_frame_counts, obs_frame_counts


def photon_density_to_flux(wave, photon_dens, filt):
    flux = 0.0
    wl_idx = 0
    for wl, resp in zip(filt.wave_data, filt.resp_data):
        while wave[wl_idx] < wl:
            flux += resp * photon_dens[wl_idx]
            wl_idx += 1

    return flux


def flux_to_magnitude(flux, filt):
    try:
        return -2.5 * math.log(flux, 10) + filt.zp
    except ValueError:
        print(f"Error on flux: {flux}")
        raise


def magnitude_to_flux(mag, filt):
    return 10 ** ((mag - filt.zp) / (-2.5))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    filters = {}
    for filt_name in ["Ukait", "Bkait", "Vkait", "Rkait", "Ikait"]:
        filters[filt_name] = snpy.fset[filt_name]

    epoch = 3
    z = 0.8
    raw_wave, raw_spec = snpy.getSED(epoch)
    avg = np.mean(raw_spec)
    raw_spec = np.array([avg for _ in raw_spec])

    # Use angstroms starting at 0.
    rest_wave = [float(val) for val in range(int(raw_wave[-1]))]
    rest_photon_dens = [0.0 for _ in rest_wave]
    rest_wave_idx = 0
    while rest_wave[rest_wave_idx] < raw_wave[0]:
        rest_wave_idx += 1

    h_erg = 1.988e-16  # Planks constant in erg*s.
    c_cm = 299792458 * 100  # c in cm/s
    hc = h_erg * c_cm

    for idx in range(len(raw_wave) - 1):
        # Solve:
        # energy = photons * integral(hc / lambda)
        # photons = energy / (hc * (log(wl_hi) - log(wl_lo)))
        photons = raw_spec[idx] / (
            hc * (math.log(raw_wave[idx + 1]) - math.log(raw_wave[idx]))
        )
        photons_per_ang = photons / (raw_wave[idx + 1] - raw_wave[idx])
        while (
            rest_wave_idx < len(rest_wave)
            and rest_wave[rest_wave_idx] < raw_wave[idx + 1]
        ):
            rest_photon_dens[rest_wave_idx] = photons_per_ang
            rest_wave_idx += 1

    obs_wave = [float(val) for val in range(2 + int(rest_wave[-1] * (1 + z)))]
    obs_photon_dens = [0.0 for _ in obs_wave]
    for idx, wl in enumerate(rest_wave):
        lo_wl = wl * (1 + z)
        hi_wl = (wl + 1) * (1 + z)
        time_dilated_photon_dens = rest_photon_dens[idx] / (1 + z)
        photons_per_ang = time_dilated_photon_dens / (hi_wl - lo_wl)

        dist = 0.0

        lo_wl_idx = int(lo_wl)
        if lo_wl_idx < lo_wl:
            obs_photon_dens[lo_wl_idx] += (lo_wl_idx + 1.0 - lo_wl) * photons_per_ang
            dist += (lo_wl_idx + 1.0 - lo_wl) * photons_per_ang
            lo_wl_idx += 1

        while lo_wl_idx + 1 <= hi_wl:
            dist += photons_per_ang
            obs_photon_dens[lo_wl_idx] += photons_per_ang
            lo_wl_idx += 1

        if hi_wl != int(hi_wl):
            dist += (hi_wl - int(hi_wl)) * photons_per_ang
            obs_photon_dens[lo_wl_idx] += (hi_wl - int(hi_wl)) * photons_per_ang

    rest_filt = filters["Bkait"]
    obs_filt = filters["Rkait"]

    real_mag = rest_filt.synth_mag(raw_wave, raw_spec)
    obs_mag = obs_filt.synth_mag(raw_wave, raw_spec, z=z)
    obs_mag_2 = obs_filt.synth_mag(
            np.array([wl * (1 + z) for wl in raw_wave]),
            np.array([spec / (1 + z) for spec in raw_spec]))
    k = snpy.kcorr.K(wave=raw_wave, spec=raw_spec, f1=rest_filt, f2=obs_filt, z=z)[0]

    print("z             :", z)
    print("real_mag      :", real_mag)
    print("obs_mag       :", obs_mag)
    print("obs_mag 2     :", obs_mag_2)
    print("k             :", k)
    k_ext = k - 2.5 * math.log(1 + z, 10)
    print("k ext         :", k - 2.5 * math.log(1 + z, 10))
    print("k-cor mag     :", obs_mag - k)
    # TODO(lpe): This seems to be the smoking gun -- k-cor mag ext produces the
    # correct value, but k_ext uses my fix for k-corrections. Review this after
    # some sleep.
    print("k-cor mag ext :", obs_mag - k_ext) # !!!
    print("k-cor mag2    :", obs_mag_2 - k)

    exit()

    snpy_rest_flux = rest_filt.response(raw_wave, raw_spec)
    print("snpy rest flux:", snpy_rest_flux)
    snpy_obs_flux = obs_filt.response(raw_wave, raw_spec, z=z)
    print("snpy obs  flux:", snpy_obs_flux)
    print("ratio:", snpy_rest_flux / snpy_obs_flux)
    #plt.scatter(rest_filt.wave_data, rest_filt.resp_data, s=1)
    #plt.scatter(obs_filt.wave_data, obs_filt.resp_data, s=1, color='red')
    plt.scatter(rest_wave, rest_photon_dens, s=1)
    plt.scatter(obs_wave, obs_photon_dens, s=1, color="red")
    plt.show()

    my_rest_flux = photon_density_to_flux(rest_wave, rest_photon_dens, rest_filt)
    my_rest_mag = flux_to_magnitude(my_rest_flux, rest_filt)
    my_obs_flux = photon_density_to_flux(obs_wave, obs_photon_dens, obs_filt)
    my_obs_mag = flux_to_magnitude(my_obs_flux, obs_filt)

    print("  my rest flux:", my_rest_flux)
    print("  my rest mag :", my_rest_mag)
    print("  my obs  flux:", my_obs_flux)
    print("  my obs  mag :", my_obs_mag)
    print(" my flux ratio:", my_rest_flux / my_obs_flux)
    # print("corrected mag    :", my_obs_mag - k)
    # print("my corrected mag :", (my_obs_mag - k) - 2.5 * math.log(1 + z, 10))

    mc_counts = frame_counts_monte_carlo(
        z=z, rest_photon_dens=rest_photon_dens, filters=filters, trials=10000
    )
    pprint(mc_counts)
    print(" mc flux ratio:", mc_counts[0][rest_filt.name] / mc_counts[1][obs_filt.name])

    print(
        "my mag (mc):",
        flux_to_magnitude(
            magnitude_to_flux(my_obs_mag, obs_filt)
            * mc_counts[0][rest_filt.name]
            / mc_counts[1][obs_filt.name],
            rest_filt,
        ),
    )

    exit()

    print(f"name,z,filt,epoch,flux,mag,pub_mag,my_mag")
    for sn in Supernova.get_all():
        pub_mag = SnInfo.all_info()[sn.name].max_mag
        my_mag = pub_mag - 2.5 * math.log(1 + sn.z, 10)
        for obs in sn.observations:
            if obs.flux <= 0.0:
                continue

            # counts = frame_counts_monte_carlo(
            #    z=sn.z, epoch=obs.epoch, trials=1000, time_dilation=True
            # )
            counts = frame_counts_deterministic(z=sn.z, epoch=obs.epoch)

            if not counts:
                continue

            rest_frame_counts, obs_frame_counts = counts
            if obs_frame_counts[obs.filt.name] == 0:
                continue

            flux = obs.flux * rest_frame_counts["Bkait"] / obs_frame_counts[obs.filt.name]

            mag = flux_to_magnitude(flux)
            print(
                f"{sn.name},{sn.z},{obs.filt.name},{obs.epoch},{flux},{mag},{pub_mag},{my_mag}"
            )
