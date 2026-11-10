"""
orbit_fit.py - fit each moon's orbit as a PROJECTED ELLIPSE (optionally with
the ephemeris depth offset dz) and draw the sky-plane orbit diagrams.

Model
-----
A moon on a circular orbit, seen at a slight tilt, traces an ellipse on the
sky around its planet:

    u =  a * sin(theta)           along the orbit's major axis
    v =  a * k * cos(theta)       along the minor axis   (k = b/a, signed)
    theta = 2*pi*t/P + phase

    xi  = x0 + u*cos(psi) - v*sin(psi)      (east offset, arcsec)
    eta = y0 + u*sin(psi) + v*cos(psi)      (north offset, arcsec)

and, if the depth offset is available, the line-of-sight component

    z   = s * a * sqrt(1 - k^2) * cos(theta)     (s = +1 or -1, chosen by the fit)

  a    semi-major axis      = the real orbital radius as seen on the sky
  k*a  semi-minor axis      = why the moon never reaches zero separation
  psi  tilt of the major axis on the sky (position angle of the orbit)
  P    orbital period

For a moon in its planet's equatorial plane, |k| = sin(B), where B is the
tilt of the ring plane towards us.

No orbital period is assumed. The period is found by scanning every period
from MIN_PERIOD_DAYS up to MAX_PERIOD_SPAN_FACTOR x the observing span, and
the best few candidates are then refined with the full non-linear model.
Competing periods that fit nearly as well are reported as aliases.

Images taken close together in time (VISIT_GAP_MINUTES) are averaged into one
"visit" before fitting, because they are not independent measurements.

dz (optional columns dz_km and dsat_km)
---------------------------------------
dz = (distance Earth-planet) - (distance Earth-moon) from an ephemeris, so it
already contains the known orbit. With USE_DZ_IN_FIT = True the period is
fitted from the sky positions AND dz, and is therefore partly informed by the
ephemeris. Set USE_DZ_IN_FIT = False for a result from the sky positions
alone (dz is then only reported as a consistency check).

Each observing year is fitted on its own. Nothing is combined across years.

Usage (in batch_fit.py):
    import orbit_fit
    angular = calculate_angular_separations(centres)
    orbit_fit.fit_and_plot_orbits(angular, OUT_DIR, None, PARENT_BODY)

(The third argument is kept only so that older calls still work; it is
ignored, because no starting period is used.)

Needs the columns written by calculate_angular_separations():
datetime, moon, sky_x_arcsec, sky_y_arcsec, separation_arcsec, error_arcsec,
sigma_dx, sigma_dy, cd1_1, cd1_2, cd2_1, cd2_2, and optionally dz_km, dsat_km.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import f as f_distribution


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum separate visits before a fit is attempted. The fit has 5 free
# parameters and each visit gives 2 numbers (3 with dz), so 4 visits is the
# bare mathematical minimum and leaves almost no redundancy.
MIN_VISITS = 6
MIN_VISITS_WITH_DZ = 4

# Images closer together than this are averaged into one visit.
VISIT_GAP_MINUTES = 30.0

# Warn if the fit rests on fewer distinct nights than this.
MIN_NIGHTS_FOR_RELIABLE_FIT = 3

# Period search (no assumed period).
MIN_PERIOD_DAYS = 0.5
MAX_PERIOD_SPAN_FACTOR = 3.0
SCAN_OVERSAMPLE = 400        # fine, because well-measured data has narrow dips
N_CANDIDATES = 6             # best periods refined with the full model
ALIAS_DELTA_CHI2 = 9.0       # alternatives within this (in noise units) are "not ruled out"

# Use dz as a third coordinate in the fit (see the module docstring).
USE_DZ_IN_FIT = True

# Also fit a constant offset (x0, y0) of the planet centre. Leave False
# unless you suspect the planet centroid is biased (rings, saturation).
FIT_CENTRE_OFFSET = False

# Draw the model curve over at least one full period in the time plots.
EXTEND_MODEL_TO_FULL_PERIOD = True

# 3D orbit diagrams (one extra figure per moon and year).
MAKE_3D_PLOTS = True

# Animation of the fitted orbits moving through the observing period, with
# each real measurement appearing on the night it was taken.
MAKE_ANIMATIONS = True
ANIMATION_FRAMES = 240            # frames per animation
ANIMATION_FPS = 24
ANIMATION_FORMATS = ("gif", "mp4")   # mp4 needs ffmpeg installed; skipped if missing

# Used ONLY to draw the 3D plot to scale when the data has no dsat_km column.
# Approximate Earth-planet distances in AU; the figure is labelled
# "approximate scale" when these are used. Replace with your own values, or
# add a dSat column to relative_positions.csv for an exact scale.
FALLBACK_PLANET_DISTANCE_AU = {"saturn": 9.0, "neptune": 29.0}

# Moons that also get an eccentric (Kepler) fit, compared with the circular
# fit. Titan is the only one whose eccentricity (0.029) is large enough to
# matter at these measurement errors; the others would be consistent with 0.
ECCENTRIC_MOONS = {"titan"}

# Equatorial radius of each planet in km, for drawing it to scale in 3D.
PLANET_RADIUS_KM = {"saturn": 60268.0, "neptune": 24764.0}

ARCSEC_PER_RAD = 206264.80624709636


# ============================================================
# MODEL
# ============================================================

def ellipse_model(t, a, k, psi, period, phase, x0=0.0, y0=0.0):
    """Projected circular orbit. Returns (xi, eta) in arcsec."""

    theta = 2 * np.pi * np.asarray(t) / period + phase
    u = a * np.sin(theta)
    v = a * k * np.cos(theta)

    xi = x0 + u * np.cos(psi) - v * np.sin(psi)
    eta = y0 + u * np.sin(psi) + v * np.cos(psi)

    return xi, eta


def depth_model(t, a, k, period, phase, sign):
    """Line-of-sight offset of the same orbit, in arcsec-equivalent."""

    theta = 2 * np.pi * np.asarray(t) / period + phase
    return sign * a * np.sqrt(max(1.0 - k * k, 0.0)) * np.cos(theta)


def _wrap(psi, phase):
    """
    (psi + pi, phase + pi) describes the same ellipse as (psi, phase).
    Report psi in [-90, 90) degrees and shift the phase to match.
    """

    m = np.floor((psi + np.pi / 2) / np.pi)
    return psi - m * np.pi, (phase - m * np.pi) % (2 * np.pi)


def ensure_distance_columns(df):
    """
    Make sure the table has dz_km and dsat_km columns.

    Accepts the names used in relative_positions.csv after the loader
    lower-cases them (for example "dz", "dz (km)", "dsat"). Values are read
    as AU if their typical size is below 10, otherwise as km.
    """

    def find(prefix):
        # "dsat" also matches dSaturn; dPlanet / dDist are accepted too.
        names = ("dsat", "dplanet", "ddist") if prefix == "dsat" else (prefix,)

        for column in df.columns:
            if str(column).lower().replace(" ", "").startswith(names):
                return column
        return None

    for target, prefix in (("dz_km", "dz"), ("dsat_km", "dsat")):
        if target in df.columns:
            continue

        source = find(prefix)

        if source is None:
            continue

        values = pd.to_numeric(df[source], errors="coerce")
        median = np.nanmedian(np.abs(values)) if values.notna().any() else np.nan
        in_au = np.isfinite(median) and median < 10
        df[target] = values * 1.495978707e8 if in_au else values
        print(f"Using column '{source}' as {target} "
              f"({'AU converted to km' if in_au else 'already km'}).")

    return df


def sky_errors(df):
    """Propagate pixel errors through the CD matrix, per sky axis."""

    needed = ["cd1_1", "cd1_2", "cd2_1", "cd2_2", "sigma_dx", "sigma_dy"]

    if not all(c in df.columns for c in needed):
        nan = np.full(len(df), np.nan)
        return nan, nan

    cd = df[needed[:4]].to_numpy(dtype=float) * 3600.0
    sdx = df["sigma_dx"].to_numpy(dtype=float, copy=True)
    sdy = df["sigma_dy"].to_numpy(dtype=float, copy=True)

    # Missing centroid errors get the typical error of the measured ones,
    # so one missing file does not remove the weights for a whole moon.
    for values in (sdx, sdy):
        finite = values[np.isfinite(values)]

        if len(finite):
            values[~np.isfinite(values)] = float(np.median(finite))

    sx = np.hypot(cd[:, 0] * sdx, cd[:, 1] * sdy)
    sy = np.hypot(cd[:, 2] * sdx, cd[:, 3] * sdy)

    return sx, sy


# ============================================================
# VISITS
# ============================================================

def combine_visits(group):
    """
    Average images taken close together into one point per visit.

    Returns a DataFrame with columns t, xi, eta, sxi, seta, z, sz, n_images.
    z is dz expressed in arcsec-equivalent (dz / km-per-arcsec), NaN if dz or
    the planet distance is missing.
    """

    group = group.sort_values("datetime")
    gaps = group["datetime"].diff().dt.total_seconds().div(60.0).fillna(np.inf)
    visit_id = (gaps > VISIT_GAP_MINUTES).cumsum()

    # The planet's distance changes during the run, and an orbit of fixed
    # size in km looks bigger on the sky when the planet is closer. To fit
    # one fixed orbit, every visit is converted to "arcsec at the reference
    # distance" (the average distance of this moon's data):
    #     angle_ref = angle_seen * (distance_that_night / reference_distance)
    # so all three coordinates are proportional to km and the fitted size
    # is a constant orbit radius.
    all_ds = (
        group["dsat_km"].to_numpy(dtype=float)
        if "dsat_km" in group else np.array([np.nan])
    )
    all_ds = all_ds[np.isfinite(all_ds)]
    d_ref = float(all_ds.mean()) if len(all_ds) else np.nan

    rows = []

    for _, v in group.groupby(visit_id):
        record = {
            "t": float(v["t"].mean()),
            "n_images": len(v),
        }

        for name, value_col, err_col in (
            ("xi", "sky_x_arcsec", "sigma_sky_x"),
            ("eta", "sky_y_arcsec", "sigma_sky_y"),
        ):
            values = v[value_col].to_numpy(dtype=float)
            errors = v[err_col].to_numpy(dtype=float)

            if np.all(np.isfinite(errors)) and np.all(errors > 0):
                w = 1.0 / errors ** 2
                record[name] = float(np.sum(w * values) / np.sum(w))
                record["s" + name] = float(1.0 / np.sqrt(np.sum(w)))
            else:
                record[name] = float(values.mean())
                record["s" + name] = np.nan

        dz = v["dz_km"].to_numpy(dtype=float) if "dz_km" in v else np.array([np.nan])
        ds = v["dsat_km"].to_numpy(dtype=float) if "dsat_km" in v else np.array([np.nan])

        if np.isfinite(ds).any() and np.isfinite(d_ref):
            factor = float(np.nanmean(ds)) / d_ref
            for key in ("xi", "eta", "sxi", "seta"):
                record[key] = record[key] * factor
        record["distance_factor"] = (
            float(np.nanmean(ds)) / d_ref
            if np.isfinite(ds).any() and np.isfinite(d_ref) else np.nan
        )

        if np.isfinite(dz).any() and np.isfinite(d_ref):
            record["z"] = float(np.nanmean(dz) / (d_ref / ARCSEC_PER_RAD))
        else:
            record["z"] = np.nan

        rows.append(record)

    return pd.DataFrame(rows)


# ============================================================
# PERIOD SEARCH
# ============================================================

def _basis(t, frequency):
    omega = 2.0 * np.pi * frequency
    return np.column_stack([np.sin(omega * t), np.cos(omega * t)])


def _linear_chi2(t, frequency, series, weights):
    """Sum of weighted squared residuals of free sin/cos fits to each series."""

    design = _basis(t, frequency)
    total = 0.0

    for values, w in zip(series, weights):
        root = np.sqrt(w)
        coeff, *_ = np.linalg.lstsq(
            design * root[:, None], values * root, rcond=None
        )
        total += float(np.sum(((values - design @ coeff) * root) ** 2))

    return total


def candidate_periods(t, series, weights):
    """
    Scan every trial period and return the best distinct candidates
    (periods in days), refined, best first.
    """

    span = float(np.ptp(t))
    f_min = 1.0 / (MAX_PERIOD_SPAN_FACTOR * span)
    f_max = 1.0 / MIN_PERIOD_DAYS
    step = 1.0 / (SCAN_OVERSAMPLE * span)

    freqs = np.arange(f_min, f_max + step, step)
    chi2 = np.array([_linear_chi2(t, f, series, weights) for f in freqs])

    padded = np.r_[np.inf, chi2, np.inf]
    minima = [
        i for i in range(len(chi2))
        if padded[i + 1] < padded[i] and padded[i + 1] <= padded[i + 2]
    ]
    minima.sort(key=lambda i: chi2[i])

    refined = []

    for i in minima[:N_CANDIDATES]:
        lo = freqs[max(i - 1, 0)]
        hi = freqs[min(i + 1, len(freqs) - 1)]
        res = minimize_scalar(
            lambda f: _linear_chi2(t, f, series, weights),
            bounds=(lo, hi), method="bounded",
        )
        refined.append((float(res.fun), 1.0 / float(res.x)))

    refined.sort()
    return [period for _, period in refined]


# ============================================================
# FIT
# ============================================================

def fit_orbit(vis):
    """
    Weighted least-squares fit of the projected ellipse to the visits.

    `vis` is the table from combine_visits(). Returns a dict, or None if the
    data cannot support a fit.
    """

    t = vis["t"].to_numpy(dtype=float)
    xi = vis["xi"].to_numpy(dtype=float)
    eta = vis["eta"].to_numpy(dtype=float)
    sxi = vis["sxi"].to_numpy(dtype=float)
    seta = vis["seta"].to_numpy(dtype=float)
    z = vis["z"].to_numpy(dtype=float)

    n = len(t)
    use_z = USE_DZ_IN_FIT and np.all(np.isfinite(z))
    n_coords = 3 if use_z else 2
    n_par = 7 if FIT_CENTRE_OFFSET else 5

    needed = MIN_VISITS_WITH_DZ if use_z else MIN_VISITS

    if n < needed or np.ptp(t) <= 0 or n_coords * n <= n_par:
        return None

    weighted = (
        np.all(np.isfinite(sxi)) and np.all(sxi > 0)
        and np.all(np.isfinite(seta)) and np.all(seta > 0)
    )

    wx = 1.0 / sxi if weighted else np.ones(n)
    wy = 1.0 / seta if weighted else np.ones(n)

    if use_z:
        sz = np.hypot(sxi, seta) / np.sqrt(2.0) if weighted else np.ones(n)
        wz = 1.0 / sz if weighted else np.ones(n)
    else:
        wz = None

    # ---- period search: every period, nothing assumed -------------------
    series = [xi, eta] + ([z] if use_z else [])
    weights = [wx ** 2, wy ** 2] + ([wz ** 2] if use_z else [])
    periods = candidate_periods(t, series, weights)

    if not periods:
        return None

    def starting_values(period):
        """Orientation, size, tilt and phase from a linear fit at this period."""

        design = _basis(t, 1.0 / period)
        cx, *_ = np.linalg.lstsq(design * wx[:, None], xi * wx, rcond=None)
        cy, *_ = np.linalg.lstsq(design * wy[:, None], eta * wy, rcond=None)

        matrix = np.array([cx, cy])
        u_mat, singular, v_t = np.linalg.svd(matrix)

        a_start = max(float(singular[0]), 1e-3)
        k_start = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
        psi_start = float(np.arctan2(u_mat[1, 0], u_mat[0, 0]))
        phase_start = float(np.arctan2(v_t[0, 1], v_t[0, 0]))

        return a_start, min(k_start, 0.99), psi_start, phase_start

    span = float(np.ptp(t))
    p_lo = MIN_PERIOD_DAYS * 0.9
    p_hi = MAX_PERIOD_SPAN_FACTOR * span * 1.1

    def unpack(p):
        if FIT_CENTRE_OFFSET:
            return p[:5], p[5], p[6]
        return p[:5], 0.0, 0.0

    lower = [0.0, -1.0, -np.pi, p_lo, -np.inf]
    upper = [np.inf, 1.0, np.pi, p_hi, np.inf]

    if FIT_CENTRE_OFFSET:
        lower += [-np.inf, -np.inf]
        upper += [np.inf, np.inf]

    signs = (1.0, -1.0) if use_z else (1.0,)

    best_by_candidate = []

    for period_start in periods:
        best_here = None

        for sign in signs:

            def residuals(p, sign=sign):
                (a, k, psi, period, phase), x0, y0 = unpack(p)
                mx, my = ellipse_model(t, a, k, psi, period, phase, x0, y0)
                parts = [(xi - mx) * wx, (eta - my) * wy]

                if use_z:
                    parts.append((z - depth_model(t, a, k, period, phase, sign)) * wz)

                return np.concatenate(parts)

            a_s, k_s, psi_s, phase_s = starting_values(period_start)

            for k_start in (k_s, -k_s):
                p0 = [a_s, k_start, psi_s, period_start, phase_s]

                if FIT_CENTRE_OFFSET:
                    p0 += [0.0, 0.0]

                p0[3] = float(np.clip(p0[3], p_lo * 1.001, p_hi * 0.999))

                try:
                    result = least_squares(
                        residuals, p0,
                        bounds=(lower, upper),
                        max_nfev=500,
                    )
                except (ValueError, RuntimeError):
                    continue

                if best_here is None or result.cost < best_here[0].cost:
                    best_here = (result, sign)

        if best_here is not None:
            best_by_candidate.append(best_here)

    if not best_by_candidate:
        return None

    best_by_candidate.sort(key=lambda item: item[0].cost)
    best, sign = best_by_candidate[0]

    p = best.x
    chi2 = float(np.sum(best.fun ** 2))
    dof = n_coords * n - n_par
    chi2_red = chi2 / dof

    cov = np.linalg.pinv(best.jac.T @ best.jac)

    if not weighted:
        # No usable centroid errors: scale by the scatter of the fit.
        cov = cov * chi2_red

    err = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # Scale for judging aliases: with trusted errors the noise unit is at
    # least 1, so a lucky tiny chi-squared cannot make aliases look unlikely.
    scale = max(chi2_red, 1.0 if weighted else 1e-12)

    alternatives = []
    next_period, next_delta = np.nan, np.inf

    for other, _ in best_by_candidate[1:]:
        other_chi2 = float(np.sum(other.fun ** 2))
        other_period = float(other.x[3])
        delta = (other_chi2 - chi2) / scale

        if abs(other_period - p[3]) / p[3] < 0.01:
            continue

        if not np.isfinite(next_delta) or delta < next_delta:
            next_period, next_delta = other_period, delta

        if delta < ALIAS_DELTA_CHI2:
            alternatives.append((other_period, delta))

    at_bound = bool(
        np.isclose(p[3], p_lo, rtol=1e-3) or np.isclose(p[3], p_hi, rtol=1e-3)
    )

    if at_bound:
        period_status = "at edge of search range"
    elif alternatives:
        period_status = "ambiguous"
    else:
        period_status = "unique"

    psi_w, phase_w = _wrap(p[2], p[4])

    # _wrap shifts the phase by m*pi, which flips cos(theta) when m is odd,
    # so the depth sign has to flip with it to describe the same orbit.
    shift = int(np.floor((p[2] + np.pi / 2) / np.pi))
    sign_wrapped = sign * (-1.0) ** shift

    return {
        "a": p[0], "a_err": err[0],
        "k": p[1], "k_err": err[1],
        "psi": psi_w, "psi_err": err[2],
        "period": p[3], "period_err": err[3],
        "phase": phase_w,
        "x0": p[5] if FIT_CENTRE_OFFSET else 0.0,
        "y0": p[6] if FIT_CENTRE_OFFSET else 0.0,
        "chi2_red": chi2_red,
        "weighted": weighted,
        "used_dz": use_z,
        "z_sign": sign_wrapped,
        "psi_raw": p[2], "phase_raw": p[4], "z_sign_raw": sign,
        "chi2": chi2, "dof": dof,
        "period_status": period_status,
        "alternatives": alternatives,
        "next_period": next_period,
        "next_delta": next_delta,
        "at_period_bound": at_bound,
    }


def kepler_E(mean_anomaly, e):
    """Solve E - e sin E = M for the eccentric anomaly (Newton iteration)."""

    m = np.mod(mean_anomaly, 2 * np.pi)
    big_e = m + e * np.sin(m)

    for _ in range(30):
        big_e = big_e - (big_e - e * np.sin(big_e) - m) / (1 - e * np.cos(big_e))

    return big_e


def ecc_model(t, a, e, psi, k, period, m0, omega, sign):
    """
    Keplerian orbit with the planet at the focus, seen at tilt k.

    Returns (xi, eta, z) in arcsec-equivalent. For e = 0 it reduces to the
    circular model above (with omega - m0 = pi/2 - phase).
    """

    mean = 2 * np.pi * np.asarray(t) / period + m0
    big_e = kepler_E(mean, e)

    p = a * (np.cos(big_e) - e)                     # towards the closest point
    q = -a * np.sqrt(1 - e * e) * np.sin(big_e)     # same sense as circular fit

    x1 = p * np.cos(omega) - q * np.sin(omega)      # along the line of nodes
    x2 = p * np.sin(omega) + q * np.cos(omega)      # across it, in the orbit plane

    xi = x1 * np.cos(psi) - k * x2 * np.sin(psi)
    eta = x1 * np.sin(psi) + k * x2 * np.cos(psi)
    z = sign * np.sqrt(max(1.0 - k * k, 0.0)) * x2

    return xi, eta, z


def fit_eccentric(vis, circ):
    """
    Fit a Kepler orbit (adds eccentricity e, the direction of closest
    approach and the phase) starting from the circular solution.

    Returns a dict, or None. The period is only allowed to move 5% from the
    circular fit, so this refines the orbit and cannot jump to an alias.
    The improvement over the circular fit is judged with an F-test, which
    does not depend on knowing the absolute size of the errors.
    """

    t = vis["t"].to_numpy(dtype=float)
    xi = vis["xi"].to_numpy(dtype=float)
    eta = vis["eta"].to_numpy(dtype=float)
    sxi = vis["sxi"].to_numpy(dtype=float)
    seta = vis["seta"].to_numpy(dtype=float)
    z = vis["z"].to_numpy(dtype=float)

    n = len(t)
    use_z = circ["used_dz"]
    n_coords = 3 if use_z else 2
    n_par = 7
    dof = n_coords * n - n_par

    # Seven free parameters need a reasonable number of visits; with fewer
    # the eccentricity is not constrained and would only be fitting noise.
    if n < 6 or dof <= 1 or (not use_z and n < 8):
        return None

    weighted = circ["weighted"]
    wx = 1.0 / sxi if weighted else np.ones(n)
    wy = 1.0 / seta if weighted else np.ones(n)

    if use_z:
        sz = np.hypot(sxi, seta) / np.sqrt(2.0) if weighted else np.ones(n)
        wz = 1.0 / sz if weighted else np.ones(n)

    period0 = circ["period"]
    lower = [0.0, 0.0, -np.inf, -1.0, period0 * 0.95, -np.inf, -np.inf]
    upper = [np.inf, 0.6, np.inf, 1.0, period0 * 1.05, np.inf, np.inf]

    best = None

    for sign in ((1.0, -1.0) if use_z else (1.0,)):

        def residuals(p, sign=sign):
            a, e, psi, k, period, m0, omega = p
            mx, my, mz = ecc_model(t, a, e, psi, k, period, m0, omega, sign)
            parts = [(xi - mx) * wx, (eta - my) * wy]

            if use_z:
                parts.append((z - mz) * wz)

            return np.concatenate(parts)

        for m0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            for e0 in (0.02, 0.08):
                omega0 = m0 + np.pi / 2 - circ["phase_raw"]
                p0 = [circ["a"], e0, circ["psi_raw"], circ["k"],
                      period0, m0, omega0]

                try:
                    result = least_squares(
                        residuals, p0, bounds=(lower, upper), max_nfev=400,
                    )
                except (ValueError, RuntimeError):
                    continue

                if best is None or result.cost < best[0].cost:
                    best = (result, sign)

    if best is None:
        return None

    result, sign = best
    p = result.x
    chi2 = float(np.sum(result.fun ** 2))
    chi2_red = chi2 / dof

    cov = np.linalg.pinv(result.jac.T @ result.jac)
    # Conservative: if the model does not fit within the stated errors,
    # inflate the parameter errors by the misfit.
    cov = cov * max(chi2_red, 1.0 if weighted else 1e-12) if weighted else cov * chi2_red
    err = np.sqrt(np.maximum(np.diag(cov), 0.0))

    delta = circ["chi2"] - chi2
    extra = n_par - 5

    if delta > 0 and chi2 > 0:
        f_value = (delta / extra) / (chi2 / dof)
        p_value = float(f_distribution.sf(f_value, extra, dof))
    else:
        p_value = 1.0

    return {
        "a": p[0], "a_err": err[0],
        "e": p[1], "e_err": err[1],
        "psi": p[2], "k": p[3],
        "period": p[4], "period_err": err[4],
        "m0": p[5], "omega": p[6], "sign": sign,
        "chi2": chi2, "chi2_red": chi2_red, "dof": dof,
        "delta_chi2": float(delta), "p_value": p_value,
    }


def ecc_xyz(ecc, t):
    """Eccentric-fit position (xi, eta, z), arcsec-equivalent."""

    return ecc_model(
        t, ecc["a"], ecc["e"], ecc["psi"], ecc["k"], ecc["period"],
        ecc["m0"], ecc["omega"], ecc["sign"],
    )


def model_from_fit(fit, t):
    return ellipse_model(
        t, fit["a"], fit["k"], fit["psi"], fit["period"], fit["phase"],
        fit["x0"], fit["y0"],
    )


# ============================================================
# PLOTS
# ============================================================

def _sky_axes_labels(ax):
    ax.set_xlabel("RA offset (arcsec)   East ←")
    ax.set_ylabel("Dec offset (arcsec)   → North")
    ax.invert_xaxis()      # east on the left, as on the sky
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)


def plot_moon_year(group, fit, moon, year, parent, origin, path):
    """2 x 2 figure: sky-plane orbit, separation, RA(t), Dec(t)."""

    t = group["t"].to_numpy()
    xi = group["sky_x_arcsec"].to_numpy()
    eta = group["sky_y_arcsec"].to_numpy()
    sep = group["separation_arcsec"].to_numpy()
    sxi = group["sigma_sky_x"].to_numpy()
    seta = group["sigma_sky_y"].to_numpy()
    ssep = group["error_arcsec"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax_sky, ax_sep, ax_xi, ax_eta = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )

    def bars(x, y, err):
        return np.where(np.isfinite(err), err, 0.0)

    # ---- sky-plane orbit ------------------------------------
    ax_sky.errorbar(
        xi, eta, xerr=bars(xi, xi, sxi), yerr=bars(eta, eta, seta),
        fmt="none", ecolor="0.6", zorder=2,
    )
    sc = ax_sky.scatter(
        xi, eta, c=t, cmap="viridis", s=36, zorder=3,
        label="Observed position",
    )
    ax_sky.plot(
        0, 0, "o", color="gold", mec="k", ms=12, zorder=4,
        label=parent.capitalize(),
    )
    cbar = fig.colorbar(sc, ax=ax_sky, fraction=0.046, pad=0.04)
    cbar.set_label("Days since first image")

    if fit is not None:
        tt = np.linspace(0, fit["period"], 600)
        ex, ey = model_from_fit(fit, tt)
        ax_sky.plot(ex, ey, "--", color="tab:red", lw=1.2, label="Fitted orbit")

        mx, my = model_from_fit(fit, t)
        for x1, y1, x2, y2 in zip(xi, eta, mx, my):
            ax_sky.plot([x1, x2], [y1, y2], "-", color="tab:red", lw=0.8)

    ecc = fit.get("ecc") if fit is not None else None

    if ecc is not None:
        ex2, ey2, _ = ecc_xyz(ecc, np.linspace(0, ecc["period"], 600))
        ax_sky.plot(ex2, ey2, "-", color="tab:orange", lw=1.4,
                    label=f"Eccentric fit (e = {ecc['e']:.3f})")

    ax_sky.set_title("Orbit on the sky (equal scale)")
    _sky_axes_labels(ax_sky)
    ax_sky.legend(fontsize=8, loc="upper right")

    # ---- model time grid ------------------------------------
    t_end = t.max()
    if fit is not None and EXTEND_MODEL_TO_FULL_PERIOD:
        t_end = max(t_end, t.min() + fit["period"])
    tg = np.linspace(t.min(), t_end, 1500)

    # ---- separation -----------------------------------------
    ax_sep.errorbar(
        t, sep, yerr=bars(sep, sep, ssep), fmt="o", capsize=3,
        label="Observed separation",
    )
    if fit is not None:
        gx, gy = model_from_fit(fit, tg)
        ax_sep.plot(tg, np.hypot(gx, gy), "-", color="tab:red", label="Model")
        ax_sep.axhline(
            fit["a"], ls=":", color="k", lw=1,
            label=f"a = {fit['a']:.1f}″ (max)",
        )
        ax_sep.axhline(
            abs(fit["k"]) * fit["a"], ls=":", color="tab:blue", lw=1,
            label=f"b = {abs(fit['k']) * fit['a']:.2f}″ (min)",
        )
    ax_sep.set_title(f"Separation from {parent.capitalize()}")
    ax_sep.set_xlabel("Days since first image")
    ax_sep.set_ylabel("Separation (arcsec)")
    ax_sep.set_ylim(bottom=0)
    ax_sep.grid(True, alpha=0.3)
    ax_sep.legend(fontsize=8)

    # ---- RA and Dec against time ----------------------------
    for ax, data, err, name, comp in (
        (ax_xi, xi, sxi, "RA offset (arcsec, east +)", 0),
        (ax_eta, eta, seta, "Dec offset (arcsec, north +)", 1),
    ):
        ax.errorbar(
            t, data, yerr=bars(data, data, err), fmt="o", capsize=3,
            label="Observed",
        )
        if fit is not None:
            ax.plot(
                tg, model_from_fit(fit, tg)[comp], "-", color="tab:red",
                label="Circular model",
            )

            if ecc is not None:
                ax.plot(
                    tg, ecc_xyz(ecc, tg)[comp], "-", color="tab:orange",
                    label="Eccentric model",
                )
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("Days since first image")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    if fit is not None:
        tilt = np.degrees(np.arcsin(min(abs(fit["k"]), 1.0)))
        title = (
            f"{moon.capitalize()} {year}:  a = {fit['a']:.2f} ± "
            f"{fit['a_err']:.2f}″,  b/a = {abs(fit['k']):.3f} "
            f"(tilt {tilt:.1f}°),  P = {fit['period']:.4f} ± "
            f"{fit['period_err']:.4f} d ({fit['period_status']}, "
            f"{'with dz' if fit['used_dz'] else 'sky only'}),  "
            f"χ²$_{{red}}$ = {fit['chi2_red']:.2f}"
        )
    else:
        title = f"{moon.capitalize()} {year}: no orbit fit (too few separate visits)"

    fig.suptitle(title, fontsize=10)
    fig.text(
        0.5, 0.005,
        f"Origin of time axis: {origin:%Y-%m-%d %H:%M}",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_orbit_3d(group, fit, moon, year, parent, origin, path):
    """
    3D orbit diagram with the planet at the centre.

    Axes (kilometres): East, North, and depth towards Earth (dz). Four
    views: 3D, the view from Earth (what the telescope sees), east against
    depth, and north against depth. The east-depth view is the closest to a
    top-down view of the orbit, because the orbit is nearly edge-on from
    Earth.
    """

    if "dz_km" not in group or not np.isfinite(group["dz_km"]).any():
        print(
            f"No 3D plot for {moon} {year}: no usable dz values "
            "(need a dz column in relative_positions.csv)."
        )
        return

    dsat = (
        group["dsat_km"].to_numpy(dtype=float)
        if "dsat_km" in group else np.array([np.nan])
    )
    dsat = dsat[np.isfinite(dsat)]

    approximate = len(dsat) == 0

    if approximate:
        au = FALLBACK_PLANET_DISTANCE_AU.get(parent)

        if au is None:
            print(
                f"No 3D plot for {moon} {year}: no dSat column and no "
                f"fallback distance for '{parent}' (add it to "
                "FALLBACK_PLANET_DISTANCE_AU or give the moon a parent in "
                "PARENT_BODY)."
            )
            return

        distance_km = au * 1.495978707e8
    else:
        distance_km = float(dsat.mean())

    km = distance_km / ARCSEC_PER_RAD     # km per arcsecond

    good = np.isfinite(group["dz_km"].to_numpy(dtype=float))
    # Each image uses the planet's distance on its own night.
    if "dsat_km" in group:
        km_each = group["dsat_km"].to_numpy(dtype=float)[good] / ARCSEC_PER_RAD
        km_each = np.where(np.isfinite(km_each), km_each, km)
    else:
        km_each = km
    obs_x = group["sky_x_arcsec"].to_numpy(dtype=float)[good] * km_each
    obs_y = group["sky_y_arcsec"].to_numpy(dtype=float)[good] * km_each
    obs_z = group["dz_km"].to_numpy(dtype=float)[good]
    obs_t = group["t"].to_numpy(dtype=float)[good]

    # Fitted orbit as a 3D circle (km).
    curve = None

    if fit is not None:
        tt = np.linspace(0, fit["period"], 600)
        cx, cy = model_from_fit(fit, tt)

        # Direction of travel along the line of sight is fixed by the fit
        # when dz was used; otherwise choose the sign that matches dz.
        if fit["used_dz"]:
            sign = fit["z_sign"]
        else:
            z_obs_arc = obs_z / km
            theta_t = obs_t
            candidates = []

            for trial in (1.0, -1.0):
                model = depth_model(
                    theta_t, fit["a"], fit["k"], fit["period"],
                    fit["phase"], trial,
                )
                candidates.append((np.sum((z_obs_arc - model) ** 2), trial))

            sign = min(candidates)[1]

        cz = depth_model(
            tt, fit["a"], fit["k"], fit["period"], fit["phase"], sign
        )
        curve = (cx * km, cy * km, cz * km)

    ecc_curve = None
    ecc = fit.get("ecc") if fit is not None else None

    if ecc is not None:
        ex, ey, ez = ecc_xyz(ecc, np.linspace(0, ecc["period"], 600))
        ecc_curve = (ex * km, ey * km, ez * km)

    # Equal-scale limits in all three directions.
    pieces = [obs_x, obs_y, obs_z]
    if curve is not None:
        pieces += list(curve)
    if ecc_curve is not None:
        pieces += list(ecc_curve)
    radius = 1.08 * max(float(np.max(np.abs(a))) for a in pieces)

    fig = plt.figure(figsize=(13, 10))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax_sky = fig.add_subplot(2, 2, 2)
    ax_top = fig.add_subplot(2, 2, 3)
    ax_side = fig.add_subplot(2, 2, 4)

    scale = 1e6   # show megametres

    # ---- 3D ----
    ax3.scatter(
        obs_x / scale, obs_y / scale, obs_z / scale,
        c=obs_t, cmap="viridis", s=34, depthshade=False, label="Observed",
    )
    if curve is not None:
        ax3.plot(
            curve[0] / scale, curve[1] / scale, curve[2] / scale,
            "--", color="tab:red", lw=1.2, label="Fitted orbit",
        )
    if ecc_curve is not None:
        ax3.plot(
            ecc_curve[0] / scale, ecc_curve[1] / scale, ecc_curve[2] / scale,
            "-", color="tab:orange", lw=1.4,
            label=f"Eccentric fit (e = {ecc['e']:.3f})",
        )
    ax3.scatter([0], [0], [0], color="gold", edgecolor="k", s=140,
                label=parent.capitalize())
    ax3.plot([0, 0], [0, 0], [0, radius / scale], color="0.4", lw=1)
    ax3.text(0, 0, radius / scale, " to Earth", fontsize=8)
    ax3.set_xlim(-radius / scale, radius / scale)
    ax3.set_ylim(-radius / scale, radius / scale)
    ax3.set_zlim(-radius / scale, radius / scale)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("East (10$^6$ km)")
    ax3.set_ylabel("North (10$^6$ km)")
    ax3.set_zlabel("Towards Earth (10$^6$ km)")
    ax3.view_init(elev=22, azim=-60)
    ax3.legend(fontsize=7, loc="upper left")
    ax3.set_title("3D orbit")

    def panel(ax, xs, ys, cxs, cys, xlabel, ylabel, title, ecc_xy=None):
        ax.scatter(xs / scale, ys / scale, c=obs_t, cmap="viridis", s=34,
                   zorder=3)
        if cxs is not None:
            ax.plot(cxs / scale, cys / scale, "--", color="tab:red", lw=1.2)
        if ecc_xy is not None:
            ax.plot(ecc_xy[0] / scale, ecc_xy[1] / scale, "-",
                    color="tab:orange", lw=1.4)
        ax.plot(0, 0, "o", color="gold", mec="k", ms=11, zorder=4)
        ax.set_xlim(-radius / scale, radius / scale)
        ax.set_ylim(-radius / scale, radius / scale)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    cxk, cyk, czk = curve if curve is not None else (None, None, None)

    ec = ecc_curve

    panel(ax_sky, obs_x, obs_y, cxk, cyk,
          "East (10$^6$ km)", "North (10$^6$ km)", "View from Earth",
          None if ec is None else (ec[0], ec[1]))
    panel(ax_top, obs_x, obs_z, cxk, czk,
          "East (10$^6$ km)", "Towards Earth (10$^6$ km)",
          "East vs depth (closest to a top-down view)",
          None if ec is None else (ec[0], ec[2]))
    panel(ax_side, obs_y, obs_z, cyk, czk,
          "North (10$^6$ km)", "Towards Earth (10$^6$ km)",
          "Side view (north vs depth)",
          None if ec is None else (ec[1], ec[2]))
    ax_top.invert_xaxis()

    title = f"{moon.capitalize()} {year}: orbit around {parent.capitalize()}"
    if fit is not None:
        title += (
            f"   (P = {fit['period']:.3f} d, {fit['period_status']}, "
            f"{'with dz' if fit['used_dz'] else 'sky only'})"
        )
    if approximate:
        title += "\napproximate scale: no dSat column, assumed distance used"

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _geometry_km(group, fit, parent):
    """
    Observed points and fitted 3D orbit of one moon, in km.

    Returns (obs_x, obs_y, obs_z, obs_t, curve, approximate) where curve is
    (x, y, z) arrays or None, and approximate says whether the fallback
    planet distance was used. Returns None if the moon has no dz values or
    no distance can be found.
    """

    if "dz_km" not in group or not np.isfinite(group["dz_km"]).any():
        return None

    dsat = (
        group["dsat_km"].to_numpy(dtype=float)
        if "dsat_km" in group else np.array([np.nan])
    )
    dsat = dsat[np.isfinite(dsat)]
    approximate = len(dsat) == 0

    if approximate:
        au = FALLBACK_PLANET_DISTANCE_AU.get(parent)

        if au is None:
            return None

        distance_km = au * 1.495978707e8
    else:
        distance_km = float(dsat.mean())

    km = distance_km / ARCSEC_PER_RAD

    good = np.isfinite(group["dz_km"].to_numpy(dtype=float))
    # Each image uses the planet's distance on its own night.
    if "dsat_km" in group:
        km_each = group["dsat_km"].to_numpy(dtype=float)[good] / ARCSEC_PER_RAD
        km_each = np.where(np.isfinite(km_each), km_each, km)
    else:
        km_each = km
    obs_x = group["sky_x_arcsec"].to_numpy(dtype=float)[good] * km_each
    obs_y = group["sky_y_arcsec"].to_numpy(dtype=float)[good] * km_each
    obs_z = group["dz_km"].to_numpy(dtype=float)[good]
    obs_t = group["t"].to_numpy(dtype=float)[good]

    curve = None
    ecc_curve = None

    if fit is not None:
        tt = np.linspace(0, fit["period"], 600)
        cx, cy = model_from_fit(fit, tt)

        if fit["used_dz"]:
            sign = fit["z_sign"]
        else:
            trials = []

            for trial in (1.0, -1.0):
                model = depth_model(
                    obs_t, fit["a"], fit["k"], fit["period"],
                    fit["phase"], trial,
                )
                trials.append((np.sum((obs_z / km - model) ** 2), trial))

            sign = min(trials)[1]

        cz = depth_model(
            tt, fit["a"], fit["k"], fit["period"], fit["phase"], sign
        )
        curve = (cx * km, cy * km, cz * km)

        if fit.get("ecc") is not None:
            ecc = fit["ecc"]
            ex, ey, ez = ecc_xyz(ecc, np.linspace(0, ecc["period"], 600))
            ecc_curve = (ex * km, ey * km, ez * km)

    return obs_x, obs_y, obs_z, obs_t, curve, approximate, ecc_curve


def plot_system_3d(groups, fits, parent, year, path):
    """
    All moons of one planet on a single 3D diagram, to scale.

    `groups` and `fits` map moon name to its table of images and its fit.
    Axes (km): East, North, depth towards Earth. A second panel shows the
    view from Earth and a third the east-depth view.
    """

    moons = [
        m for m in groups
        if PARENT_BODY_LOOKUP.get(m) == parent
    ]

    pieces = {}
    approximate = False

    for moon in moons:
        geometry = _geometry_km(groups[moon], fits.get(moon), parent)

        if geometry is not None:
            pieces[moon] = geometry
            approximate = approximate or geometry[5]

    if not pieces:
        print(
            f"No 3D system plot for {parent} {year}: no moon has usable "
            "dz values."
        )
        return

    arrays = []

    for obs_x, obs_y, obs_z, _, curve, _, ecc_c in pieces.values():
        arrays += [obs_x, obs_y, obs_z]
        if curve is not None:
            arrays += list(curve)
        if ecc_c is not None:
            arrays += list(ecc_c)

    radius = 1.08 * max(float(np.max(np.abs(a))) for a in arrays)
    scale = 1e6

    colours = plt.cm.tab10(np.arange(len(pieces)) % 10)

    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax_sky = fig.add_subplot(2, 4, 3)
    ax_top = fig.add_subplot(2, 4, 4)
    ax_note = fig.add_subplot(2, 2, 4)
    ax_note.axis("off")

    # Planet drawn to scale.
    r_planet = PLANET_RADIUS_KM.get(parent, 0.0) / scale
    if r_planet > 0:
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
        ax3.plot_surface(
            r_planet * np.cos(u) * np.sin(v),
            r_planet * np.sin(u) * np.sin(v),
            r_planet * np.cos(v),
            color="gold", alpha=0.9, linewidth=0,
        )

    for colour, (moon, geom) in zip(colours, pieces.items()):
        obs_x, obs_y, obs_z, _, curve, _, ecc_c = geom

        ax3.scatter(
            obs_x / scale, obs_y / scale, obs_z / scale,
            color=colour, s=22, depthshade=False, label=moon.capitalize(),
        )
        ax_sky.scatter(obs_x / scale, obs_y / scale, color=colour, s=14)
        ax_top.scatter(obs_x / scale, obs_z / scale, color=colour, s=14)

        if curve is not None:
            ax3.plot(
                curve[0] / scale, curve[1] / scale, curve[2] / scale,
                "--", color=colour, lw=1,
            )
            ax_sky.plot(curve[0] / scale, curve[1] / scale, "--",
                        color=colour, lw=0.9)
            ax_top.plot(curve[0] / scale, curve[2] / scale, "--",
                        color=colour, lw=0.9)

        if ecc_c is not None:
            # Same curve as in the moon's own plot: circular fit dashed,
            # eccentric fit solid.
            ax3.plot(ecc_c[0] / scale, ecc_c[1] / scale, ecc_c[2] / scale,
                     "-", color=colour, lw=1.6)
            ax_sky.plot(ecc_c[0] / scale, ecc_c[1] / scale, "-",
                        color=colour, lw=1.3)
            ax_top.plot(ecc_c[0] / scale, ecc_c[2] / scale, "-",
                        color=colour, lw=1.3)

    ax3.plot([0, 0], [0, 0], [0, radius / scale], color="0.4", lw=1)
    ax3.text(0, 0, radius / scale, " to Earth", fontsize=8)
    lim = radius / scale
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_zlim(-lim, lim)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("East (10$^6$ km)")
    ax3.set_ylabel("North (10$^6$ km)")
    ax3.set_zlabel("Towards Earth (10$^6$ km)")
    ax3.view_init(elev=22, azim=-60)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_title(f"{parent.capitalize()} system in 3D")

    for ax, title, xlabel, ylabel in (
        (ax_sky, "From Earth", "East", "North"),
        (ax_top, "East vs depth", "East", "Towards Earth"),
    ):
        ax.plot(0, 0, "o", color="gold", mec="k", ms=8, zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"{xlabel} (10$^6$ km)", fontsize=8)
        ax.set_ylabel(f"{ylabel} (10$^6$ km)", fontsize=8)
        ax.tick_params(labelsize=7)

    ax_top.invert_xaxis()

    lines = []
    for moon in pieces:
        fit = fits.get(moon)
        if fit is None:
            lines.append(f"{moon.capitalize()}: not fitted (points only)")
        else:
            lines.append(
                f"{moon.capitalize()}: P = {fit['period']:.3f} d "
                f"({fit['period_status']})"
            )
            ecc = fit.get("ecc")

            if ecc is not None:
                lines.append(
                    f"   eccentric fit: P = {ecc['period']:.3f} d, "
                    f"e = {ecc['e']:.3f}  (solid line; dashed = circular)"
                )

    ax_note.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9,
                 family="monospace")

    title = f"{parent.capitalize()} system, {year}"
    if approximate:
        title += "  (approximate scale: assumed planet distance)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _moon_track(group, fit, parent):
    """
    Everything needed to animate one moon, in km:
      position(t_days)   fitted position at time t (days from the moon's first image)
      curve              one full orbit (x, y, z)
      obs                (t_days, x, y, z) of the real measurements
      origin             datetime of the moon's first image
    Returns None if the moon has no fit or no distance information.
    """

    geometry = _geometry_km(group, fit, parent)

    if geometry is None or fit is None:
        return None

    obs_x, obs_y, obs_z, obs_t, curve, _, ecc_curve = geometry

    if "dsat_km" in group and np.isfinite(group["dsat_km"]).any():
        km = float(np.nanmean(group["dsat_km"])) / ARCSEC_PER_RAD
    else:
        km = FALLBACK_PLANET_DISTANCE_AU[parent] * 1.495978707e8 / ARCSEC_PER_RAD

    ecc = fit.get("ecc")

    if fit["used_dz"]:
        sign = fit["z_sign"]
    else:
        z_obs_arc = obs_z / km
        trials = [
            (np.sum((z_obs_arc - depth_model(
                obs_t, fit["a"], fit["k"], fit["period"], fit["phase"], tr
            )) ** 2), tr)
            for tr in (1.0, -1.0)
        ]
        sign = min(trials)[1]

    def position(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))

        if ecc is not None:
            x, y, z = ecc_xyz(ecc, t)
        else:
            x, y = model_from_fit(fit, t)
            z = depth_model(t, fit["a"], fit["k"], fit["period"],
                            fit["phase"], sign)

        return x * km, y * km, z * km

    period = ecc["period"] if ecc is not None else fit["period"]
    shown_curve = ecc_curve if ecc_curve is not None else curve

    return {
        "position": position,
        "curve": shown_curve,
        "obs": (obs_t, obs_x, obs_y, obs_z),
        "origin": group["datetime"].min(),
        "period": period,
        "ecc": ecc is not None,
        "status": fit["period_status"],
    }


def make_orbit_animation(groups, fits, parent, year, path_stem):
    """
    Animate the fitted orbits of one planet's moons through the observing
    period (written as <path_stem>.gif and/or .mp4).

    Left: view from Earth (east vs north). Right: east vs depth (the nearest
    thing to a view from above, since Earth looks almost along the ring
    plane). Solid dots are the fitted moons, a short trail shows where each
    has just been, faint rings are the full orbits, and the small markers are
    the real measurements, which appear on the night they were taken.
    """

    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import shutil

    tracks = {}

    for moon in sorted(groups):
        if PARENT_BODY_LOOKUP.get(moon) != parent:
            continue

        track = _moon_track(groups[moon], fits.get(moon), parent)

        if track is not None:
            tracks[moon] = track

    if not tracks:
        print(f"No animation for {parent} {year}: no moon has a usable fit with dz.")
        return

    start = min(tr["origin"] for tr in tracks.values())
    stop = max(groups[m]["datetime"].max() for m in tracks)
    span = (stop - start).total_seconds() / 86400.0
    t_start, t_end = -0.5, span + 0.5
    times = np.linspace(t_start, t_end, ANIMATION_FRAMES)

    # Per-moon offset from the shared clock, and measurement times on it.
    for tr in tracks.values():
        tr["offset"] = (tr["origin"] - start).total_seconds() / 86400.0
        tr["obs_clock"] = tr["obs"][0] + tr["offset"]

    reach = 1.1 * max(
        float(np.max(np.abs(np.concatenate([tr["curve"][0], tr["curve"][2]]))))
        for tr in tracks.values()
    )
    reach_y = 1.1 * max(
        float(np.max(np.abs(tr["curve"][1]))) for tr in tracks.values()
    )
    scale, unit = (1e6, "10$^6$ km") if reach > 5e5 else (1e3, "10$^3$ km")

    palette = plt.get_cmap("tab10")
    colours = {m: palette(i % 10) for i, m in enumerate(sorted(tracks))}

    fig, (ax_sky, ax_top) = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    for ax in (ax_sky, ax_top):
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.scatter([0], [0], s=180, color="gold", edgecolor="k", zorder=5)

    ax_sky.set_xlim(-reach / scale, reach / scale)
    ax_sky.set_ylim(-max(reach_y, 0.25 * reach) / scale,
                    max(reach_y, 0.25 * reach) / scale)
    ax_sky.set_xlabel(f"East ({unit})")
    ax_sky.set_ylabel(f"North ({unit})")
    ax_sky.set_title("View from Earth")

    ax_top.set_xlim(reach / scale, -reach / scale)      # east to the left
    ax_top.set_ylim(-reach / scale, reach / scale)
    ax_top.set_xlabel(f"East ({unit})")
    ax_top.set_ylabel(f"Towards Earth ({unit})")
    ax_top.set_title("East vs depth (closest to a view from above)")

    artists = {}

    for moon, tr in tracks.items():
        c = colours[moon]
        cx, cy, cz = tr["curve"]
        ax_sky.plot(cx / scale, cy / scale, "-", color=c, lw=0.8, alpha=0.35)
        ax_top.plot(cx / scale, cz / scale, "-", color=c, lw=0.8, alpha=0.35)

        label = moon.capitalize() + (" (eccentric)" if tr["ecc"] else "")

        artists[moon] = {
            "dot_sky": ax_sky.plot([], [], "o", color=c, ms=9, mec="k",
                                   zorder=6, label=label)[0],
            "dot_top": ax_top.plot([], [], "o", color=c, ms=9, mec="k",
                                   zorder=6)[0],
            "trail_sky": ax_sky.plot([], [], "-", color=c, lw=2.2, alpha=0.8)[0],
            "trail_top": ax_top.plot([], [], "-", color=c, lw=2.2, alpha=0.8)[0],
            "obs_sky": ax_sky.plot([], [], "D", color=c, ms=5.5, mec="k",
                                   mew=0.6, zorder=7, alpha=0.9)[0],
            "obs_top": ax_top.plot([], [], "D", color=c, ms=5.5, mec="k",
                                   mew=0.6, zorder=7, alpha=0.9)[0],
        }

    ax_sky.legend(loc="upper right", fontsize=8, framealpha=0.9)
    clock = fig.suptitle("", fontsize=12)
    fig.text(
        0.5, 0.015,
        "Dots: fitted orbits.  Diamonds: real measurements, shown from the "
        "night they were taken.  Lines: the last part of each path.",
        ha="center", fontsize=8, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))

    def draw(frame):
        now = times[frame]
        stamp = start + pd.Timedelta(days=float(now))
        clock.set_text(f"{parent.capitalize()} system, {year}   "
                       f"{stamp:%d %b %Y  %H:%M}")

        for moon, tr in tracks.items():
            a = artists[moon]
            t_moon = now - tr["offset"]
            x, y, z = tr["position"](t_moon)
            a["dot_sky"].set_data(x / scale, y / scale)
            a["dot_top"].set_data(x / scale, z / scale)

            back = np.linspace(t_moon - min(0.18 * tr["period"], 4.0), t_moon, 30)
            bx, by, bz = tr["position"](back)
            a["trail_sky"].set_data(bx / scale, by / scale)
            a["trail_top"].set_data(bx / scale, bz / scale)

            seen = tr["obs_clock"] <= now
            _, ox, oy, oz = tr["obs"]
            a["obs_sky"].set_data(ox[seen] / scale, oy[seen] / scale)
            a["obs_top"].set_data(ox[seen] / scale, oz[seen] / scale)

        return []

    animation = FuncAnimation(fig, draw, frames=len(times), blit=False)

    for fmt in ANIMATION_FORMATS:
        target = f"{path_stem}.{fmt}"

        try:
            if fmt == "gif":
                animation.save(target, writer=PillowWriter(fps=ANIMATION_FPS), dpi=90)
            elif fmt == "mp4":
                if shutil.which("ffmpeg") is None:
                    print("Skipping mp4: ffmpeg is not installed "
                          "(the gif was still made).")
                    continue
                animation.save(target, writer=FFMpegWriter(fps=ANIMATION_FPS), dpi=110)
            else:
                continue

            print(f"Saved animation {target}")
        except Exception as error:      # an animation must never stop the fit
            print(f"Could not write {target}: {error}")

    plt.close(fig)


def plot_system_view(year_df, fits, parent, year, path):
    """All moons of one planet on a single sky-plane diagram."""

    moons = [m for m in year_df["moon"].unique()
             if PARENT_BODY_LOOKUP.get(m) == parent]

    if not moons:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    colours = plt.cm.tab10(np.arange(len(moons)) % 10)

    ax.plot(0, 0, "o", color="gold", mec="k", ms=14, zorder=5,
            label=parent.capitalize())

    for colour, moon in zip(colours, moons):
        g = year_df.loc[year_df["moon"] == moon]
        ax.plot(
            g["sky_x_arcsec"], g["sky_y_arcsec"], "o", color=colour,
            ms=5, label=moon.capitalize(), zorder=4,
        )

        fit = fits.get(moon)
        if fit is not None:
            tt = np.linspace(0, fit["period"], 600)
            ex, ey = model_from_fit(fit, tt)
            ax.plot(ex, ey, "--", color=colour, lw=1)

    ax.set_title(f"{parent.capitalize()} system on the sky, {year}")
    _sky_axes_labels(ax)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# DRIVER
# ============================================================

PARENT_BODY_LOOKUP = {}

FIT_COLUMNS = [
    "moon", "year", "parent", "n_images", "n_visits", "n_nights", "status",
    "used_dz_in_fit", "a_arcsec", "a_err_arcsec", "a_km", "a_err_km",
    "b_arcsec", "b_over_a", "b_over_a_err",
    "tilt_deg", "psi_deg", "psi_err_deg",
    "period_days", "period_err_days", "period_status",
    "next_best_period_days", "next_best_delta_chi2",
    "phase_rad", "chi2_red",
    "ecc_e", "ecc_e_err", "ecc_period_days", "ecc_period_err_days",
    "ecc_a_arcsec", "ecc_a_km", "ecc_a_err_km", "ecc_chi2_red", "ecc_delta_chi2",
    "ecc_p_value",
]


def fit_and_plot_orbits(df, out_dir, initial_periods=None, parent_body=None):
    """
    Fit and plot every moon, for every calendar year, independently.

    `initial_periods` is ignored (kept so older calls still work): no
    starting period is used.

    Writes:
        <out_dir>/orbit_fits.csv, orbit_fits_<year>.csv
        <out_dir>/orbit_plots/<year>/<moon>_orbit.png
        <out_dir>/orbit_plots/<year>/<moon>_orbit3d.png
        <out_dir>/orbit_plots/<year>/<planet>_system.png
        <out_dir>/orbit_plots/<year>/<planet>_system3d.png
    """

    global PARENT_BODY_LOOKUP
    parent_body = dict(parent_body or {})
    PARENT_BODY_LOOKUP = parent_body

    out_dir = Path(out_dir)
    plot_root = out_dir / "orbit_plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "sky_x_arcsec", "sky_y_arcsec"])
    df["year"] = df["datetime"].dt.year
    df["sigma_sky_x"], df["sigma_sky_y"] = sky_errors(df)
    df = ensure_distance_columns(df)

    if MAKE_3D_PLOTS and "dz_km" not in df.columns:
        print(
            "Note: no dz column found, so no 3D plots will be made. "
            "Add dz to relative_positions.csv and make sure your pipeline "
            "keeps that column."
        )

    all_records = []

    for year, year_df in df.groupby("year"):
        year_dir = plot_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        fits = {}
        groups = {}
        year_records = []

        for moon, group in year_df.groupby("moon"):
            group = group.sort_values("datetime").copy()

            origin = group["datetime"].min()
            group["t"] = (
                (group["datetime"] - origin).dt.total_seconds() / 86400.0
            )

            n_nights = int(group["datetime"].dt.normalize().nunique())
            parent = parent_body.get(moon, "planet")

            visits = combine_visits(group)
            fit = fit_orbit(visits)

            if fit is not None and moon in ECCENTRIC_MOONS:
                fit["ecc"] = fit_eccentric(visits, fit)

            fits[moon] = fit
            groups[moon] = group

            record = {
                "moon": moon, "year": year, "parent": parent,
                "n_images": len(group), "n_visits": len(visits),
                "n_nights": n_nights,
            }

            if fit is None:
                record["status"] = "not fitted: too few separate visits"
                print(
                    f"{moon} {year}: no fit; {len(visits)} separate visit(s) "
                    f"from {len(group)} images."
                )
            else:
                status = "fitted"

                if n_nights < MIN_NIGHTS_FOR_RELIABLE_FIT:
                    status += f" (only {n_nights} night(s): treat with care)"
                    print(
                        f"Warning: {moon} {year}: {len(group)} images on "
                        f"only {n_nights} night(s)."
                    )

                if fit["at_period_bound"]:
                    status += " (period at edge of search range)"
                    print(f"Warning: {moon} {year}: period at edge of search range.")

                if fit["period_status"] == "ambiguous":
                    others = ", ".join(
                        f"{p:.3f} d" for p, _ in fit["alternatives"][:3]
                    )
                    print(
                        f"Warning: {moon} {year}: other periods fit nearly as "
                        f"well ({others}). More visits at different spacings "
                        "are needed."
                    )

                dsat = group["dsat_km"].to_numpy(dtype=float) if "dsat_km" in group else np.array([np.nan])
                dsat = dsat[np.isfinite(dsat)]
                km_per_arcsec = (
                    float(dsat.mean()) / ARCSEC_PER_RAD if len(dsat) else np.nan
                )

                record.update({
                    "status": status,
                    "used_dz_in_fit": fit["used_dz"],
                    "a_arcsec": fit["a"],
                    "a_err_arcsec": fit["a_err"],
                    "a_km": fit["a"] * km_per_arcsec,
                    "a_err_km": fit["a_err"] * km_per_arcsec,
                    "b_arcsec": abs(fit["k"]) * fit["a"],
                    "b_over_a": abs(fit["k"]),
                    "b_over_a_err": fit["k_err"],
                    "tilt_deg": np.degrees(np.arcsin(min(abs(fit["k"]), 1))),
                    "psi_deg": np.degrees(fit["psi"]),
                    "psi_err_deg": np.degrees(fit["psi_err"]),
                    "period_days": fit["period"],
                    "period_err_days": fit["period_err"],
                    "period_status": fit["period_status"],
                    "next_best_period_days": fit["next_period"],
                    "next_best_delta_chi2": fit["next_delta"],
                    "phase_rad": fit["phase"],
                    "chi2_red": fit["chi2_red"],
                })

                ecc = fit.get("ecc")

                if ecc is not None:
                    record.update({
                        "ecc_e": ecc["e"],
                        "ecc_e_err": ecc["e_err"],
                        "ecc_period_days": ecc["period"],
                        "ecc_period_err_days": ecc["period_err"],
                        "ecc_a_arcsec": ecc["a"],
                        "ecc_a_km": ecc["a"] * km_per_arcsec,
                        "ecc_a_err_km": ecc["a_err"] * km_per_arcsec,
                        "ecc_chi2_red": ecc["chi2_red"],
                        "ecc_delta_chi2": ecc["delta_chi2"],
                        "ecc_p_value": ecc["p_value"],
                    })

                    verdict = (
                        "significantly better than circular"
                        if ecc["p_value"] < 0.05
                        else "not significantly better than circular"
                    )
                    print(
                        f"   {moon} {year} eccentric fit: e = {ecc['e']:.4f} "
                        f"+/- {ecc['e_err']:.4f}, P = {ecc['period']:.3f} d, "
                        f"chi2_red {fit['chi2_red']:.1f} -> "
                        f"{ecc['chi2_red']:.1f}, F-test p = "
                        f"{ecc['p_value']:.3g} ({verdict})"
                    )

                print(
                    f"{moon} {year}: a = {fit['a']:.2f}\", "
                    f"b/a = {abs(fit['k']):.3f}, "
                    f"P = {fit['period']:.4f} d ({fit['period_status']}, "
                    f"{'with dz' if fit['used_dz'] else 'sky only'}), "
                    f"chi2_red = {fit['chi2_red']:.2f}"
                )

            year_records.append(record)

            plot_moon_year(
                group, fit, moon, year, parent, origin,
                year_dir / f"{moon}_orbit.png",
            )

            if MAKE_3D_PLOTS:
                plot_orbit_3d(
                    group, fit, moon, year, parent, origin,
                    year_dir / f"{moon}_orbit3d.png",
                )

        for parent in sorted(set(parent_body.get(m) for m in fits)):
            if parent:
                plot_system_view(
                    year_df, fits, parent, year,
                    year_dir / f"{parent}_system.png",
                )

                if MAKE_3D_PLOTS:
                    plot_system_3d(
                        groups, fits, parent, year,
                        year_dir / f"{parent}_system3d.png",
                    )

                if MAKE_ANIMATIONS:
                    make_orbit_animation(
                        groups, fits, parent, year,
                        year_dir / f"{parent}_orbits_animation",
                    )

        year_table = pd.DataFrame(year_records).reindex(columns=FIT_COLUMNS)
        year_table.to_csv(out_dir / f"orbit_fits_{year}.csv", index=False)
        all_records.extend(year_records)

    pd.DataFrame(all_records).reindex(columns=FIT_COLUMNS).to_csv(
        out_dir / "orbit_fits.csv", index=False
    )

    print(f"Orbit plots saved in {plot_root}")
    print(f"Orbit fit summary saved to {out_dir / 'orbit_fits.csv'}")