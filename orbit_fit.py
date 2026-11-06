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

# Used ONLY to draw the 3D plot to scale when the data has no dsat_km column.
# Approximate Earth-planet distances in AU; the figure is labelled
# "approximate scale" when these are used. Replace with your own values, or
# add a dSat column to relative_positions.csv for an exact scale.
FALLBACK_PLANET_DISTANCE_AU = {"saturn": 9.0, "neptune": 29.0}

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
        for column in df.columns:
            if str(column).lower().replace(" ", "").startswith(prefix):
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

        if np.isfinite(dz).any() and np.isfinite(ds).any():
            record["z"] = float(np.nanmean(dz) / (np.nanmean(ds) / ARCSEC_PER_RAD))
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
        "z_sign": sign,
        "period_status": period_status,
        "alternatives": alternatives,
        "next_period": next_period,
        "next_delta": next_delta,
        "at_period_bound": at_bound,
    }


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
                label="Model",
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
    obs_x = group["sky_x_arcsec"].to_numpy(dtype=float)[good] * km
    obs_y = group["sky_y_arcsec"].to_numpy(dtype=float)[good] * km
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

    # Equal-scale limits in all three directions.
    pieces = [obs_x, obs_y, obs_z]
    if curve is not None:
        pieces += list(curve)
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

    def panel(ax, xs, ys, cxs, cys, xlabel, ylabel, title):
        ax.scatter(xs / scale, ys / scale, c=obs_t, cmap="viridis", s=34,
                   zorder=3)
        if cxs is not None:
            ax.plot(cxs / scale, cys / scale, "--", color="tab:red", lw=1.2)
        ax.plot(0, 0, "o", color="gold", mec="k", ms=11, zorder=4)
        ax.set_xlim(-radius / scale, radius / scale)
        ax.set_ylim(-radius / scale, radius / scale)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    cxk, cyk, czk = curve if curve is not None else (None, None, None)

    panel(ax_sky, obs_x, obs_y, cxk, cyk,
          "East (10$^6$ km)", "North (10$^6$ km)", "View from Earth")
    panel(ax_top, obs_x, obs_z, cxk, czk,
          "East (10$^6$ km)", "Towards Earth (10$^6$ km)",
          "View from above (east vs depth)")
    panel(ax_side, obs_y, obs_z, cyk, czk,
          "North (10$^6$ km)", "Towards Earth (10$^6$ km)",
          "Side view (north vs depth)")
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
    obs_x = group["sky_x_arcsec"].to_numpy(dtype=float)[good] * km
    obs_y = group["sky_y_arcsec"].to_numpy(dtype=float)[good] * km
    obs_z = group["dz_km"].to_numpy(dtype=float)[good]
    obs_t = group["t"].to_numpy(dtype=float)[good]

    curve = None

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

    return obs_x, obs_y, obs_z, obs_t, curve, approximate


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

    for obs_x, obs_y, obs_z, _, curve, _ in pieces.values():
        arrays += [obs_x, obs_y, obs_z]
        if curve is not None:
            arrays += list(curve)

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
        obs_x, obs_y, obs_z, _, curve, _ = geom

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

    ax_note.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=9,
                 family="monospace")

    title = f"{parent.capitalize()} system, {year}"
    if approximate:
        title += "  (approximate scale: assumed planet distance)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
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

        year_table = pd.DataFrame(year_records).reindex(columns=FIT_COLUMNS)
        year_table.to_csv(out_dir / f"orbit_fits_{year}.csv", index=False)
        all_records.extend(year_records)

    pd.DataFrame(all_records).reindex(columns=FIT_COLUMNS).to_csv(
        out_dir / "orbit_fits.csv", index=False
    )

    print(f"Orbit plots saved in {plot_root}")
    print(f"Orbit fit summary saved to {out_dir / 'orbit_fits.csv'}")