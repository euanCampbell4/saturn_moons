"""
Saturn's mass, mean density and Roche limit from the fitted moon orbits,
compared with published values.

Reads   results/orbit_fits.csv        (written by orbit_fit.py)
Writes  results/physics_summary.txt   the numbers, ready to quote
        results/physics_moons.csv     one row per moon fit
        results/kepler_third_law.png  a vs P with Kepler's law, plus residuals

Run:    python saturn_physics.py
Needs:  numpy, pandas, matplotlib

What it does
  1. Kepler's third law for each moon:  GM = 4 pi^2 a^3 / P^2
     with the error from the fitted a and P (the fit's own errors, which are
     already scaled up when the model fits worse than the stated errors).
  2. Combines the moons (weighted mean). If the moons disagree by more than
     their errors say, the error is increased to match (Birge ratio).
  3. Tests Kepler's law itself: fits log a = alpha log P + beta and checks
     alpha = 2/3.
  4. Mean density of Saturn from the measured mass and the published
     volumetric radius, and the Roche limit (fluid and rigid) for ring
     particles and for each moon.
  5. Compares everything with published values. Published numbers are used
     ONLY for comparison and for rejecting aliased periods (see PERIOD_GATE).

Neptune: if a Triton fit is present it gives Neptune's mass the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
FITS_FILE = Path("results/orbit_fits.csv")
OUT_DIR = Path("results")

ONLY_UNIQUE_PERIODS = True      # skip fits whose period the code flagged as ambiguous
PERIOD_GATE = 0.05              # reject a fit whose period differs from the published
                                # period of that moon by more than this fraction
                                # (catches aliases such as the Enceladus 3.78 d fit).
                                # Set to None to switch the gate off.
EXCLUDE = set()                 # e.g. {("enceladus", 2026)} to drop a moon-year by hand
CORRECT_FOR_MOON_MASS = True    # Kepler gives G(M + m); remove the moon's own mass m
USE_ECCENTRIC_WHEN_AVAILABLE = True   # Titan: use the eccentric fit's a and P

# ----------------------------------------------------------------------
# PUBLISHED VALUES (for comparison only)
# ----------------------------------------------------------------------
G = 6.67430e-11                       # m^3 kg^-1 s^-2 (CODATA 2018)
DAY = 86400.0

LIT = {
    "saturn": {
        "mass_kg": 5.68317e26,        # Wikipedia / NASA fact sheet value
        "r_eq_km": 60268.0,
        "r_mean_km": 58232.0,         # volumetric mean radius
        "density": 687.1,             # kg/m^3
    },
    "neptune": {
        "mass_kg": 1.02413e26,
        "r_eq_km": 24764.0,
        "r_mean_km": 24622.0,
        "density": 1638.0,
    },
}

# semi-major axis (km) and period (days), PDS Rings Node satellite table;
# Triton from standard references
LIT_ORBITS = {
    "mimas": (185537.0, 0.942),
    "enceladus": (238037.0, 1.370218),
    "tethys": (294672.0, 1.887802),
    "dione": (377400.0, 2.736916),
    "rhea": (527070.0, 4.517503),
    "titan": (1221865.0, 15.945448),
    "triton": (354759.0, 5.876854),
}
LIT_TITAN_E = 0.0288

# moon masses (kg), used only for the small correction G(M+m) -> GM
MOON_MASS_KG = {
    "enceladus": 1.0802e20, "tethys": 6.1745e20, "dione": 1.0955e21,
    "rhea": 2.3065e21, "titan": 1.3452e23, "triton": 2.139e22,
}

# mean densities (kg/m^3) for the per-moon Roche limit; ice for ring particles
MOON_DENSITY = {
    "water ice (ring particles)": 920.0,
    "enceladus": 1609.0, "tethys": 984.0, "dione": 1478.0,
    "rhea": 1236.0, "titan": 1881.0,
}

# edges of Saturn's main rings (km from the centre), PDS Rings Node
A_RING_OUTER_KM = 136770.0
F_RING_KM = 140224.0

PARENT = {m: "saturn" for m in
          ("mimas", "enceladus", "tethys", "dione", "rhea", "titan")}
PARENT["triton"] = "neptune"


def load():
    df = pd.read_csv(FITS_FILE)
    rows = []
    for _, r in df.iterrows():
        moon, year = str(r["moon"]).lower(), int(r["year"])
        label = f"{moon} {year}"

        if pd.isna(r.get("period_days")) or pd.isna(r.get("a_km")):
            rows.append(dict(moon=moon, year=year, used=False,
                             why="no fit (too few visits)"))
            continue

        a, a_err = float(r["a_km"]), float(r["a_err_km"])
        p, p_err = float(r["period_days"]), float(r["period_err_days"])
        kind = "circular"

        if (USE_ECCENTRIC_WHEN_AVAILABLE and pd.notna(r.get("ecc_period_days"))
                and pd.notna(r.get("ecc_a_km")) and pd.notna(r.get("ecc_a_err_km"))):
            a, a_err = float(r["ecc_a_km"]), float(r["ecc_a_err_km"])
            p, p_err = float(r["ecc_period_days"]), float(r["ecc_period_err_days"])
            kind = f"eccentric (e={r['ecc_e']:.3f})"

        why = ""
        if ONLY_UNIQUE_PERIODS and "unique" not in str(r.get("period_status", "")):
            why = f"period flagged '{r.get('period_status')}'"
        if (moon, year) in EXCLUDE:
            why = "excluded by hand"
        if not why and PERIOD_GATE is not None and moon in LIT_ORBITS:
            ratio = p / LIT_ORBITS[moon][1]
            if abs(ratio - 1) > PERIOD_GATE:
                why = (f"period {p:.3f} d is {100 * (ratio - 1):+.0f}% from the published "
                       f"{LIT_ORBITS[moon][1]:.3f} d (probably an alias)")
        if not why and not (np.isfinite(a_err) and np.isfinite(p_err) and a_err > 0 and p_err > 0):
            why = "no usable error estimate"

        rows.append(dict(moon=moon, year=year, parent=PARENT.get(moon, str(r["parent"])),
                         a_km=a, a_err_km=a_err, p_days=p, p_err_days=p_err,
                         kind=kind, used=(why == ""), why=why,
                         period_status=r.get("period_status", "")))
    return pd.DataFrame(rows)


def gm_from_orbit(a_km, a_err, p_days, p_err):
    """Kepler III: GM (km^3/s^2) = 4 pi^2 a^3 / P^2, with its error."""
    p_s = p_days * DAY
    gm = 4 * np.pi ** 2 * a_km ** 3 / p_s ** 2
    rel = np.sqrt((3 * a_err / a_km) ** 2 + (2 * p_err / p_days) ** 2)
    return gm, gm * rel


def weighted_mean(values, errors):
    w = 1.0 / errors ** 2
    mean = np.sum(w * values) / np.sum(w)
    err = 1.0 / np.sqrt(np.sum(w))
    n = len(values)
    chi2 = np.sum(((values - mean) / errors) ** 2)
    birge = np.sqrt(chi2 / (n - 1)) if n > 1 else np.nan
    err_scaled = err * max(birge, 1.0) if n > 1 else err
    return mean, err, err_scaled, chi2, birge


def kepler_slope(a, a_err, p, p_err):
    """Weighted fit of log a = alpha log P + beta (Kepler III says alpha = 2/3)."""
    x, y = np.log(p), np.log(a)
    sy = np.sqrt((a_err / a) ** 2 + (2.0 / 3.0 * p_err / p) ** 2)   # x error folded in at slope 2/3
    w = 1.0 / sy ** 2
    sw, swx, swy = w.sum(), (w * x).sum(), (w * y).sum()
    swxx, swxy = (w * x * x).sum(), (w * x * y).sum()
    det = sw * swxx - swx ** 2
    alpha = (sw * swxy - swx * swy) / det
    beta = (swxx * swy - swx * swxy) / det
    cov_aa = sw / det
    resid = y - (alpha * x + beta)
    chi2 = np.sum((resid / sy) ** 2)
    dof = len(x) - 2
    scale = max(np.sqrt(chi2 / dof), 1.0) if dof > 0 else 1.0
    return alpha, np.sqrt(cov_aa) * scale, beta, chi2, dof


def pct(x, ref):
    return 100.0 * (x - ref) / ref


def planet_section(df, planet, lines, make_plot=None):
    used = df[(df["parent"] == planet) & (df["used"])].copy()
    lit = LIT[planet]
    gm_lit = G * lit["mass_kg"] / 1e9            # km^3/s^2

    lines.append(f"\n{'=' * 70}\n{planet.upper()}\n{'=' * 70}")
    if used.empty:
        lines.append("No usable moon fits.")
        return None

    gms, gm_errs = [], []
    for _, r in used.iterrows():
        gm, gm_err = gm_from_orbit(r.a_km, r.a_err_km, r.p_days, r.p_err_days)
        if CORRECT_FOR_MOON_MASS and r.moon in MOON_MASS_KG:
            gm -= G * MOON_MASS_KG[r.moon] / 1e9
        gms.append(gm)
        gm_errs.append(gm_err)
    used["gm"], used["gm_err"] = gms, gm_errs
    used["gm_dev_pct"] = pct(used["gm"], gm_lit)
    used["gm_dev_sigma"] = (used["gm"] - gm_lit) / used["gm_err"]

    lines.append("\nPer-moon Kepler III (GM in km^3/s^2):")
    lines.append(f"{'moon':<10}{'year':<6}{'a (km)':>12}{'P (d)':>10}{'GM':>16}"
                 f"{'error %':>9}{'vs lit %':>10}{'sigma':>7}  fit")
    for _, r in used.iterrows():
        lines.append(f"{r.moon:<10}{r.year:<6}{r.a_km:>12,.0f}{r.p_days:>10.4f}"
                     f"{r.gm:>16,.0f}{100 * r.gm_err / r.gm:>9.2f}"
                     f"{r.gm_dev_pct:>+10.2f}{r.gm_dev_sigma:>+7.1f}  {r.kind}")

    excluded = df[(df["parent"] == planet) & (~df["used"]) & df["why"].astype(bool)]
    if len(excluded):
        lines.append("\nNot used:")
        for _, r in excluded.iterrows():
            lines.append(f"  {r.moon} {r.year}: {r.why}")

    gm, err, err_scaled, chi2, birge = weighted_mean(used["gm"].to_numpy(),
                                                      used["gm_err"].to_numpy())
    mass = gm * 1e9 / G
    mass_err = err_scaled * 1e9 / G

    lines.append(f"\nCombined ({len(used)} fits, weighted mean):")
    lines.append(f"  GM   = {gm:,.0f} +/- {err_scaled:,.0f} km^3/s^2"
                 f"   (published {gm_lit:,.0f};  {pct(gm, gm_lit):+.2f}%, "
                 f"{(gm - gm_lit) / err_scaled:+.1f} sigma)")
    lines.append(f"  Mass = ({mass:.4e} +/- {mass_err:.1e}) kg"
                 f"   (published {lit['mass_kg']:.5e} kg)")
    if len(used) > 1:
        lines.append(f"  Consistency between moons: chi2 = {chi2:.1f} for {len(used) - 1} dof"
                     f"  (Birge ratio {birge:.2f};"
                     f" {'error scaled up to match the scatter' if birge > 1 else 'moons agree within errors'})")

    # Kepler's law itself
    if len(used) >= 3 and used["moon"].nunique() >= 3:
        alpha, alpha_err, beta, c2, dof = kepler_slope(
            used["a_km"].to_numpy(), used["a_err_km"].to_numpy(),
            used["p_days"].to_numpy(), used["p_err_days"].to_numpy())
        lines.append(f"\nKepler's third law test: a ~ P^alpha")
        lines.append(f"  alpha = {alpha:.4f} +/- {alpha_err:.4f}   (Kepler: 2/3 = 0.6667;"
                     f" {(alpha - 2 / 3) / alpha_err:+.1f} sigma)")
    else:
        alpha = None
        lines.append("\nKepler's law exponent: need at least 3 different moons.")

    # Density and Roche limit
    r_mean, r_eq = lit["r_mean_km"], lit["r_eq_km"]
    rho = mass / (4.0 / 3.0 * np.pi * (r_mean * 1e3) ** 3)
    rho_err = rho * mass_err / mass
    lines.append(f"\nMean density (from the measured mass and the published "
                 f"volumetric mean radius {r_mean:,.0f} km):")
    lines.append(f"  rho = {rho:.0f} +/- {rho_err:.0f} kg/m^3   (published {lit['density']:.0f};"
                 f" {pct(rho, lit['density']):+.1f}%)")

    if planet == "saturn":
        lines.append("\nRoche limit  d = k * R * (rho_planet / rho_satellite)^(1/3)"
                     "   with R = equatorial radius")
        lines.append("  fluid body k = 2.44, rigid body k = 1.26.  Uncertainty comes from the mass only.")
        lines.append(f"{'satellite':<28}{'rho':>7}{'fluid (km)':>13}{'rigid (km)':>12}   in R_Saturn (fluid)")
        for name, rho_s in MOON_DENSITY.items():
            fac = (rho / rho_s) ** (1 / 3)
            fl, rg = 2.44 * r_eq * fac, 1.26 * r_eq * fac
            err = fl * rho_err / rho / 3
            lines.append(f"{name:<28}{rho_s:>7.0f}{fl:>13,.0f}{rg:>12,.0f}   {fl / r_eq:.2f}  (+/- {err:,.0f} km)")
        fac = (rho / MOON_DENSITY['water ice (ring particles)']) ** (1 / 3)
        fluid_ice = 2.44 * r_eq * fac
        lines.append(f"\n  For ice, fluid Roche limit {fluid_ice:,.0f} km = {fluid_ice / r_eq:.2f} R_Saturn."
                     f"  Published: A ring outer edge {A_RING_OUTER_KM:,.0f} km, F ring {F_RING_KM:,.0f} km;"
                     f" the F ring is {pct(F_RING_KM, fluid_ice):+.1f}% from this limit.")

        lines.append("\nMeasured orbit sizes in Saturn radii, against the fluid Roche limit for ice:")
        for _, r in used.iterrows():
            lines.append(f"  {r.moon:<10}{r.year}: a = {r.a_km / r_eq:6.2f} R_Saturn"
                         f"   ({r.a_km / fluid_ice:5.2f} x Roche limit)")

    # Titan's eccentricity
    ecc = df[(df["moon"] == "titan") & (df["used"])]
    lines.append("")

    # Comparison with published orbits
    lines.append("Orbit sizes and periods against published values:")
    lines.append(f"{'moon':<10}{'year':<6}{'a (km)':>12}{'published':>12}{'diff %':>8}"
                 f"{'P (d)':>10}{'published':>11}{'diff %':>8}")
    for _, r in used.iterrows():
        if r.moon in LIT_ORBITS:
            la, lp = LIT_ORBITS[r.moon]
            lines.append(f"{r.moon:<10}{r.year:<6}{r.a_km:>12,.0f}{la:>12,.0f}{pct(r.a_km, la):>+8.2f}"
                         f"{r.p_days:>10.4f}{lp:>11.4f}{pct(r.p_days, lp):>+8.2f}")

    return used, gm, err_scaled, gm_lit, alpha


def main():
    if not FITS_FILE.exists():
        raise SystemExit(f"{FITS_FILE} not found. Run the orbit fit first.")

    df = load()
    lines = ["MEASURED PLANET PROPERTIES FROM MOON ORBITS",
             "Published values are used only for comparison and to reject aliased periods."]
    results = {}

    for planet in ("saturn", "neptune"):
        if (df["parent"] == planet).any() if "parent" in df else False:
            out = planet_section(df, planet, lines)
            if out is not None:
                results[planet] = out

    # Titan eccentricity (from orbit_fits.csv)
    raw = pd.read_csv(FITS_FILE)
    t = raw[(raw["moon"] == "titan") & raw["ecc_e"].notna()]
    if len(t):
        lines.append(f"\n{'=' * 70}\nTITAN ECCENTRICITY\n{'=' * 70}")
        for _, r in t.iterrows():
            lines.append(f"  {int(r['year'])}: e = {r['ecc_e']:.4f} +/- {r['ecc_e_err']:.4f}"
                         f"   (published {LIT_TITAN_E}; {(r['ecc_e'] - LIT_TITAN_E) / r['ecc_e_err']:+.1f} sigma;"
                         f" F-test vs circular p = {r['ecc_p_value']:.2g})")
        lines.append("  Caution: the depth values (dz) come from an ephemeris, so this is a"
                     " consistency check, not an independent measurement.")

    text = "\n".join(lines)
    print(text)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "physics_summary.txt").write_text(text + "\n")
    df.to_csv(OUT_DIR / "physics_moons.csv", index=False)

    # Plot: Kepler's third law
    if "saturn" in results:
        used, gm, gm_err, gm_lit, alpha = results["saturn"]
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.3]})
        pgrid = np.logspace(np.log10(0.8), np.log10(20), 100)
        ax.loglog(pgrid, (gm * (pgrid * DAY) ** 2 / (4 * np.pi ** 2)) ** (1 / 3), "r-",
                  label=f"Measured GM = {gm:,.0f} km$^3$/s$^2$")
        ax.loglog(pgrid, (gm_lit * (pgrid * DAY) ** 2 / (4 * np.pi ** 2)) ** (1 / 3), "k--",
                  lw=0.9, label=f"Published GM = {gm_lit:,.0f}")
        colours = plt.get_cmap("tab10")
        for i, (moon, g) in enumerate(used.groupby("moon")):
            ax.errorbar(g["p_days"], g["a_km"], yerr=g["a_err_km"], xerr=g["p_err_days"],
                        fmt="o", color=colours(i), label=moon.capitalize(), capsize=3)
            ax2.errorbar(g["p_days"], g["gm_dev_pct"], yerr=100 * g["gm_err"] / gm_lit,
                         fmt="o", color=colours(i), capsize=3)
        ax.set_ylabel("Orbit radius a (km)")
        ax.set_title("Kepler's third law: Saturn's moons")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel("Period (days)")
        ax2.set_ylabel("GM vs published (%)")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "kepler_third_law.png", dpi=150)
        plt.close(fig)
        print(f"\nPlot saved: {OUT_DIR / 'kepler_third_law.png'}")

    print(f"Summary saved: {OUT_DIR / 'physics_summary.txt'}")


if __name__ == "__main__":
    main()
