"""
orbit_fit.py - fit each moon's orbit as a PROJECTED ELLIPSE and draw the
sky-plane orbit diagrams.

Why this replaces the |sin| fit
-------------------------------
A moon on a circular orbit, seen at a slight tilt, traces an ellipse on the
sky around its planet:

    u =  a * sin(theta)           along the orbit's major axis
    v =  a * k * cos(theta)       along the minor axis   (k = b/a, signed)
    theta = 2*pi*t/P + phase

    xi  = x0 + u*cos(psi) - v*sin(psi)      (east offset, arcsec)
    eta = y0 + u*sin(psi) + v*cos(psi)      (north offset, arcsec)

  a    semi-major axis      = the real orbital radius as seen on the sky
  k*a  semi-minor axis      = why the moon never reaches zero separation
  psi  tilt of the major axis on the sky (position angle of the orbit)
  P    orbital period

For a moon in its planet's equatorial plane, |k| = sin(B), where B is the
tilt of the ring plane towards us. The separation from the planet is
sqrt(xi^2 + eta^2), which has a minimum of |k|*a and a maximum of a. In the
old model "offset + amplitude*|sin|", the semi-major axis was
offset + amplitude, NOT amplitude, and the minimum was the offset.

The fit uses BOTH sky coordinates (xi, eta) at once, so it also knows which
side of the planet the moon is on and which way it is moving. That pins down
the phase and the period much better than separation alone.

Each observing year is fitted on its own. Nothing is combined across years.

Usage (in batch_fit.py):
    import orbit_fit
    ...
    angular = calculate_angular_separations(centres)
    plot_angular_separations(angular)
    orbit_fit.fit_and_plot_orbits(
        angular, OUT_DIR, INITIAL_PERIODS, PARENT_BODY
    )

Needs the columns already written by calculate_angular_separations():
datetime, moon, sky_x_arcsec, sky_y_arcsec, separation_arcsec,
error_arcsec, sigma_dx, sigma_dy, cd1_1, cd1_2, cd2_1, cd2_2.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum images in a year before a fit is attempted.
MIN_OBSERVATIONS = 6

# Warn if the fit rests on fewer distinct nights than this.
MIN_NIGHTS_FOR_RELIABLE_FIT = 3

# Soft Gaussian prior on the period, as a fraction of the literature value
# (0.02 = 2 %). Set to None to let the period float within +/-50 %.
# With only a few nights, keep the prior: the period is otherwise
# poorly constrained.
PERIOD_PRIOR_FRACTION = 0.02

# Also fit a constant offset (x0, y0) of the planet centre. Leave False
# unless you suspect the planet centroid is biased (rings, saturation).
FIT_CENTRE_OFFSET = False

# Draw the model curve over at least one full period in the time plots.
EXTEND_MODEL_TO_FULL_PERIOD = True


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


def _wrap(psi, phase):
    """
    (psi + pi, phase + pi) describes the same ellipse as (psi, phase).
    Report psi in [-90, 90) degrees and shift the phase to match.
    """

    m = np.floor((psi + np.pi / 2) / np.pi)
    return psi - m * np.pi, (phase - m * np.pi) % (2 * np.pi)


def sky_errors(df):
    """Propagate pixel errors through the CD matrix, per sky axis."""

    needed = ["cd1_1", "cd1_2", "cd2_1", "cd2_2", "sigma_dx", "sigma_dy"]

    if not all(c in df.columns for c in needed):
        nan = np.full(len(df), np.nan)
        return nan, nan

    cd = df[needed[:4]].to_numpy(dtype=float) * 3600.0
    sdx = df["sigma_dx"].to_numpy(dtype=float)
    sdy = df["sigma_dy"].to_numpy(dtype=float)

    sx = np.hypot(cd[:, 0] * sdx, cd[:, 1] * sdy)
    sy = np.hypot(cd[:, 2] * sdx, cd[:, 3] * sdy)

    return sx, sy


# ============================================================
# FIT
# ============================================================

def fit_orbit(t, xi, eta, sxi, seta, period0):
    """
    Weighted least-squares fit of the projected ellipse to (xi, eta).

    Returns a dict, or None if the data cannot support a fit.
    Several starting points are tried because the problem has local minima.
    """

    n = len(t)
    n_par = 7 if FIT_CENTRE_OFFSET else 5

    if n < MIN_OBSERVATIONS or np.ptp(t) <= 0 or 2 * n <= n_par:
        return None

    weighted = (
        np.all(np.isfinite(sxi)) and np.all(sxi > 0)
        and np.all(np.isfinite(seta)) and np.all(seta > 0)
    )

    wx = 1.0 / sxi if weighted else np.ones(n)
    wy = 1.0 / seta if weighted else np.ones(n)

    # Starting orientation: principal axis of the points about the planet.
    s00 = np.sum(xi * xi)
    s11 = np.sum(eta * eta)
    s01 = np.sum(xi * eta)
    psi0 = 0.5 * np.arctan2(2 * s01, s00 - s11)
    a0 = max(np.max(np.hypot(xi, eta)), 1e-3)

    if PERIOD_PRIOR_FRACTION:
        period_starts = period0 * np.array([0.99, 1.0, 1.01])
        p_lo, p_hi = period0 * 0.5, period0 * 1.5
    else:
        period_starts = period0 * np.array([0.8, 1.0, 1.2])
        p_lo, p_hi = period0 * 0.5, period0 * 1.5

    def unpack(p):
        if FIT_CENTRE_OFFSET:
            return p[:5], p[5], p[6]
        return p[:5], 0.0, 0.0

    def residuals(p):
        (a, k, psi, period, phase), x0, y0 = unpack(p)
        mx, my = ellipse_model(t, a, k, psi, period, phase, x0, y0)

        r = np.concatenate([(xi - mx) * wx, (eta - my) * wy])

        if PERIOD_PRIOR_FRACTION:
            r = np.append(
                r, (period - period0) / (PERIOD_PRIOR_FRACTION * period0)
            )
        return r

    lower = [0.0, -1.0, -np.pi, p_lo, -np.inf]
    upper = [np.inf, 1.0, np.pi, p_hi, np.inf]

    if FIT_CENTRE_OFFSET:
        lower += [-np.inf, -np.inf]
        upper += [np.inf, np.inf]

    best = None

    for period_start in period_starts:
        for phase_start in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            for k_start in (0.05, -0.05):
                p0 = [a0, k_start, psi0, period_start, phase_start]

                if FIT_CENTRE_OFFSET:
                    p0 += [0.0, 0.0]

                try:
                    result = least_squares(
                        residuals, p0,
                        bounds=(lower, upper),
                        max_nfev=2000,
                    )
                except (ValueError, RuntimeError):
                    continue

                if best is None or result.cost < best.cost:
                    best = result

    if best is None:
        return None

    p = best.x
    chi2 = float(np.sum(best.fun[: 2 * n] ** 2))
    dof = 2 * n - n_par
    chi2_red = chi2 / dof

    cov = np.linalg.pinv(best.jac.T @ best.jac)

    if not weighted:
        # No usable centroid errors: scale by the scatter of the fit.
        cov = cov * chi2_red

    err = np.sqrt(np.maximum(np.diag(cov), 0.0))

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
        "at_period_bound": bool(
            np.isclose(p[3], p_lo, rtol=1e-3)
            or np.isclose(p[3], p_hi, rtol=1e-3)
        ),
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
    ax.set_xlabel("RA offset (arcsec)   East \u2190")
    ax.set_ylabel("Dec offset (arcsec)   \u2192 North")
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
            label=f"a = {fit['a']:.1f}\u2033 (max)",
        )
        ax_sep.axhline(
            abs(fit["k"]) * fit["a"], ls=":", color="tab:blue", lw=1,
            label=f"b = {abs(fit['k']) * fit['a']:.2f}\u2033 (min)",
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
            f"{moon.capitalize()} {year}:  a = {fit['a']:.2f} \u00b1 "
            f"{fit['a_err']:.2f}\u2033,  b/a = {abs(fit['k']):.3f} "
            f"(tilt {tilt:.1f}\u00b0),  P = {fit['period']:.4f} \u00b1 "
            f"{fit['period_err']:.4f} d,  \u03c7\u00b2$_{{red}}$ = "
            f"{fit['chi2_red']:.2f}"
        )
    else:
        title = (
            f"{moon.capitalize()} {year}: no orbit fit "
            f"(need \u2265 {MIN_OBSERVATIONS} images)"
        )

    fig.suptitle(title, fontsize=11)
    fig.text(
        0.5, 0.005,
        f"Origin of time axis: {origin:%Y-%m-%d %H:%M}",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=160)
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
    "moon", "year", "parent", "n_observations", "n_nights", "status",
    "a_arcsec", "a_err_arcsec", "b_arcsec", "b_over_a", "b_over_a_err",
    "tilt_deg", "psi_deg", "psi_err_deg",
    "period_days", "period_err_days", "phase_rad", "chi2_red",
]


def fit_and_plot_orbits(df, out_dir, initial_periods, parent_body):
    """
    Fit and plot every moon, for every calendar year, independently.

    Writes:
        <out_dir>/orbit_fits.csv, orbit_fits_<year>.csv
        <out_dir>/orbit_plots/<year>/<moon>_orbit.png
        <out_dir>/orbit_plots/<year>/<planet>_system.png
    """

    global PARENT_BODY_LOOKUP
    PARENT_BODY_LOOKUP = dict(parent_body)

    out_dir = Path(out_dir)
    plot_root = out_dir / "orbit_plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "sky_x_arcsec", "sky_y_arcsec"])
    df["year"] = df["datetime"].dt.year
    df["sigma_sky_x"], df["sigma_sky_y"] = sky_errors(df)

    all_records = []

    for year, year_df in df.groupby("year"):
        year_dir = plot_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        fits = {}
        year_records = []

        for moon, group in year_df.groupby("moon"):
            group = group.sort_values("datetime").copy()

            origin = group["datetime"].min()
            group["t"] = (
                (group["datetime"] - origin).dt.total_seconds() / 86400.0
            )

            n_nights = int(group["datetime"].dt.normalize().nunique())
            parent = parent_body.get(moon, "planet")

            period0 = initial_periods.get(
                moon, max(np.ptp(group["t"].to_numpy()), 1.0)
            )

            fit = fit_orbit(
                group["t"].to_numpy(),
                group["sky_x_arcsec"].to_numpy(),
                group["sky_y_arcsec"].to_numpy(),
                group["sigma_sky_x"].to_numpy(),
                group["sigma_sky_y"].to_numpy(),
                period0,
            )

            fits[moon] = fit

            record = {
                "moon": moon, "year": year, "parent": parent,
                "n_observations": len(group), "n_nights": n_nights,
            }

            if fit is None:
                record["status"] = "not fitted: insufficient data"
                print(f"{moon} {year}: no fit; {len(group)} observations.")
            else:
                status = "fitted"

                if n_nights < MIN_NIGHTS_FOR_RELIABLE_FIT:
                    status += f" (only {n_nights} night(s): treat with care)"
                    print(
                        f"Warning: {moon} {year}: {len(group)} images on "
                        f"only {n_nights} night(s)."
                    )

                if fit["at_period_bound"]:
                    status += " (period at bound)"
                    print(f"Warning: {moon} {year}: period hit its bound.")

                record.update({
                    "status": status,
                    "a_arcsec": fit["a"],
                    "a_err_arcsec": fit["a_err"],
                    "b_arcsec": abs(fit["k"]) * fit["a"],
                    "b_over_a": abs(fit["k"]),
                    "b_over_a_err": fit["k_err"],
                    "tilt_deg": np.degrees(np.arcsin(min(abs(fit["k"]), 1))),
                    "psi_deg": np.degrees(fit["psi"]),
                    "psi_err_deg": np.degrees(fit["psi_err"]),
                    "period_days": fit["period"],
                    "period_err_days": fit["period_err"],
                    "phase_rad": fit["phase"],
                    "chi2_red": fit["chi2_red"],
                })

                print(
                    f"{moon} {year}: a = {fit['a']:.2f}\" , "
                    f"b/a = {abs(fit['k']):.3f}, "
                    f"P = {fit['period']:.4f} d, "
                    f"chi2_red = {fit['chi2_red']:.2f}"
                )

            year_records.append(record)

            plot_moon_year(
                group, fit, moon, year, parent, origin,
                year_dir / f"{moon}_orbit.png",
            )

        for parent in sorted(set(parent_body.get(m) for m in fits)):
            if parent:
                plot_system_view(
                    year_df, fits, parent, year,
                    year_dir / f"{parent}_system.png",
                )

        year_table = pd.DataFrame(year_records, columns=FIT_COLUMNS)
        year_table.to_csv(out_dir / f"orbit_fits_{year}.csv", index=False)
        all_records.extend(year_records)

    pd.DataFrame(all_records, columns=FIT_COLUMNS).to_csv(
        out_dir / "orbit_fits.csv", index=False
    )

    print(f"Orbit plots saved in {plot_root}")
    print(f"Orbit fit summary saved to {out_dir / 'orbit_fits.csv'}")