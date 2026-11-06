"""
Automated Saturn and Neptune moon astrometry pipeline.

Input layout
------------
data/
    relative_positions.csv
    2026-10-10/
        saturn_img0009_x.csv
        saturn_img0009_y.csv
        titan_img0009_x.csv
        titan_img0009_y.csv
        neptune_img0010_x.csv
        neptune_img0010_y.csv
        triton_img0010_x.csv
        triton_img0010_y.csv

Profile filename:
    <object>_<image>_<axis>.<extension>

Examples:
    titan_img0009_x.txt
    saturn_img0009_y.csv
    neptune_img0010_x.txt
    triton_img0010_y.csv

Relative-position CSV columns:
    Data, Date, Time, CD1_1, CD1_2, CD2_1, CD2_2, dX, dY, dZ, dSat
    (optional: dZ = ephemeris distance to planet minus distance to moon;
     dSat = distance Earth-to-planet; AU or km)

Outputs
-------
results/centres.csv
results/angular_separations.csv
results/angular_separations_<year>.csv      one file per observing year
results/orbit_fits_<year>.csv               one file per observing year
results/orbit_fits.csv                      summary of the above (has a year column)
results/plots/
results/angular_plots/<year>/<moon>_orbit.png

Each observing year is fitted completely on its own: data from different
years are never combined in a fit.

Run from the repository root:
    python batch_fit.py

Requirements:
    pip install numpy pandas scipy matplotlib

Important assumptions
---------------------
1. CD values are in degrees per pixel.
2. dX/dY are offsets relative to the relevant parent planet:
   Saturn for Saturn's moons, Neptune for Triton.
3. All offsets and Gaussian centroid positions use the same
   pixel coordinate system.
4. Centroid errors are independent Gaussian errors.
5. The CD matrix uncertainty is negligible.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUT_DIR = Path("results")

RELATIVE_FILE = DATA_DIR / "relative_positions.csv"

CENTRE_PLOT_DIR = OUT_DIR / "plots"
ANGULAR_PLOT_DIR = OUT_DIR / "angular_plots"

SLOPED_BACKGROUND = False
START_PIXEL = 0

# Extend this dictionary if more satellites are added.
PARENT_BODY = {
    "titan": "saturn",
    "rhea": "saturn",
    "tethys": "saturn",
    "dione": "saturn",
    "enceladus": "saturn",
    "triton": "neptune",
}

# ---- Orbit fit settings -------------------------------------------------
# No orbital period is assumed. The fit scans every period between
# MIN_PERIOD_DAYS and MAX_PERIOD_SPAN_FACTOR x (the year's observing span).
MIN_PERIOD_DAYS = 0.5
MAX_PERIOD_SPAN_FACTOR = 3.0
PERIOD_SCAN_OVERSAMPLE = 400      # scan steps per resolvable frequency interval
                                  # (fine, because well-measured data has very narrow chi-squared dips)

# Images closer together in time than this are averaged into one "visit",
# because they are not independent measurements of the orbit.
VISIT_GAP_MINUTES = 30.0

# A fit has 5 free parameters (period + 2 x 2 ellipse coefficients) and each
# visit gives 2 numbers, so 4 visits is the bare mathematical minimum and
# leaves almost no redundancy: with a free period the fit then simply picks
# whichever period matches, and the answer is not trustworthy. 6 visits
# (7 degrees of freedom) is the smallest number that gives a meaningful test.
MIN_VISITS_FOR_FIT = 6

# With dz as a third coordinate each visit gives 3 numbers, so fewer visits
# are enough (6 linear parameters + period = 7 unknowns; 4 visits = 12 numbers).
MIN_VISITS_WITH_DZ = 4

# A competing period within this many "scale units" of chi-squared of the
# best one is reported as not ruled out (ambiguous).
ALIAS_DELTA_CHI2 = 9.0

# Use the ephemeris depth offset dz as a third coordinate in the orbit fit
# (needs dz and dSat in the CSV). The period then comes from sky positions
# AND dz together, so it is partly informed by the ephemeris. Set False to
# fit the period from the sky positions alone (dz is then only a check).
USE_DZ_IN_FIT = True

# Which orbit fit to run. False: the fit built into this file (linear period
# scan, centred ellipse). True: the separate orbit_fit.py module (physical
# ellipse model with tilt and orientation, plus system plots). Run only one,
# because both write results/orbit_fits.csv.
USE_ORBIT_FIT_MODULE = True

N_BOOTSTRAP = 300                 # resamples used for the ellipse-axis errors

# relative_positions.csv may contain two optional distance columns:
#   dz    = (distance Earth-to-planet) - (distance Earth-to-moon), from an
#           ephemeris. Positive means the moon is closer to Earth than its planet.
#   dsat  = distance Earth-to-planet (e.g. JPL Horizons "delta"), used to turn
#           arcseconds into km.
# Units of both: "auto" treats values whose median magnitude is below 10 as AU
# and larger values as km.
DIST_UNIT = "auto"                # "auto", "au" or "km"

# Used for the km conversion and the dz fit when the CSV has no dsat column.
# One Earth-planet distance in AU per planet (use JPL Horizons "delta" for a
# typical date), or None. A single number is also accepted.
PLANET_DISTANCE_AU = None          # e.g. {"saturn": 9.0, "neptune": 29.0}

# Print additional diagnostics.
VERBOSE = True


# ============================================================
# GAUSSIAN PROFILE FITTING
# ============================================================

def gauss(x, amplitude, centre, sigma, background):
    return (
        amplitude
        * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
        + background
    )


def gauss_slope(x, amplitude, centre, sigma, background, slope):
    return gauss(
        x, amplitude, centre, sigma, background
    ) + slope * (x - centre)


def load_profile(path):
    """Read ASCII counts or pixel/count pairs."""

    rows = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                values = [
                    float(value)
                    for value in line.replace(",", " ").split()
                ]
            except ValueError:
                # Ignore headers and non-numeric lines.
                continue

            if values:
                rows.append(values)

    if not rows:
        raise ValueError("No numeric data found")

    if len(rows) == 1:
        y = np.asarray(rows[0], dtype=float)

    elif all(len(row) == 1 for row in rows):
        y = np.asarray([row[0] for row in rows], dtype=float)

    elif all(len(row) == 2 for row in rows):
        array = np.asarray(rows, dtype=float)
        return array[:, 0], array[:, 1]

    else:
        raise ValueError(
            "Expected one-column counts or two columns "
            "containing pixel and counts"
        )

    x = np.arange(len(y), dtype=float) + START_PIXEL
    return x, y


def fit_gaussian(x, y):
    """Fit a Gaussian and return its model, parameters and errors."""

    model = gauss_slope if SLOPED_BACKGROUND else gauss

    background = np.median(y)
    amplitude = y.max() - background

    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("No positive peak above the background")

    initial = [
        amplitude,
        x[np.argmax(y)],
        2.0,
        background,
    ]

    if SLOPED_BACKGROUND:
        initial.append(0.0)

    parameters, covariance = curve_fit(
        model,
        x,
        y,
        p0=initial,
        maxfev=20000,
    )

    with np.errstate(invalid="ignore"):
        parameter_errors = np.sqrt(np.diag(covariance))

    centre = parameters[1]

    if not np.isfinite(centre):
        raise ValueError("Gaussian centre is not finite")

    return model, parameters, parameter_errors


def fit_all_profiles():
    """Fit all profile files and write results/centres.csv."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CENTRE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []

    for path in sorted(DATA_DIR.glob("*/*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        parts = path.stem.split("_")

        # Accept files named object_image_x or object_image_y.
        if len(parts) != 3 or parts[2].lower() not in ("x", "y"):
            continue

        object_name = parts[0].lower()
        image = parts[1].lower()
        axis = parts[2].lower()
        date = path.parent.name

        try:
            x, y = load_profile(path)
            model, parameters, errors = fit_gaussian(x, y)

            centre = parameters[1]
            centre_error = errors[1]
            fwhm = 2.354820045 * abs(parameters[2])

            records.append({
                "date": date,
                "moon": object_name,
                "image": image,
                "axis": axis,
                "centre_px": centre,
                "centre_err_px": centre_error,
                "fwhm_px": fwhm,
            })

            xx = np.linspace(x.min(), x.max(), 400)

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.plot(x, y, "k.", label="Measured profile")
            ax.plot(
                xx,
                model(xx, *parameters),
                "r-",
                label=f"Centre = {centre:.3f} +/- "
                      f"{centre_error:.3f} px",
            )

            ax.set_title(f"{date}: {path.stem}")
            ax.set_xlabel("Pixel")
            ax.set_ylabel("Counts")
            ax.legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(
                CENTRE_PLOT_DIR / f"{date}_{path.stem}.png",
                dpi=130,
            )
            plt.close(fig)

        except Exception as error:
            failures.append((path, str(error)))

    columns = [
        "date", "moon", "image", "axis",
        "centre_px", "centre_err_px", "fwhm_px",
    ]

    centres = pd.DataFrame(records, columns=columns)
    centres.to_csv(OUT_DIR / "centres.csv", index=False)

    print(f"Fitted {len(centres)} Gaussian profiles.")
    print(f"Saved {OUT_DIR / 'centres.csv'}")

    if failures:
        print(f"\nFailed to fit {len(failures)} profile(s):")
        for path, reason in failures:
            print(f"  {path}: {reason}")

    return centres


# ============================================================
# RELATIVE-POSITION DATA
# ============================================================

def load_relative_positions():
    """Load and normalise the relative-position CSV."""

    if not RELATIVE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {RELATIVE_FILE}. "
            "Place relative_positions.csv inside the data folder."
        )

    df = pd.read_csv(RELATIVE_FILE)
    df.columns = df.columns.str.strip().str.lower()

    required = {
        "data", "date", "time",
        "cd1_1", "cd1_2", "cd2_1", "cd2_2",
        "dx", "dy",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(sorted(missing))
        )

    # Extract the moon/object and image ID.
    # Example: triton_img0010 -> triton, img0010
    parsed = df["data"].astype(str).str.strip().str.lower().str.extract(
        r"^(?P<moon>.+?)_(?P<image>img[^_]+)$"
    )

    malformed = parsed.isna().any(axis=1)

    if malformed.any():
        values = df.loc[malformed, "data"].tolist()
        raise ValueError(
            f"Could not parse Data values: {values[:10]}"
        )

    df["moon"] = parsed["moon"]
    df["image"] = parsed["image"]

    # Supports dates such as 10/10/2026 and 2026-10-10.
    parsed_dates = pd.to_datetime(
        df["date"], dayfirst=True, errors="raise"
    )

    df["date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"].astype(str).str.strip(),
        errors="raise",
    )

    numeric_columns = [
        "cd1_1", "cd1_2", "cd2_1", "cd2_2", "dx", "dy",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Optional distance columns (see DIST_UNIT).
    def to_km(column):
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index)

        values = pd.to_numeric(df[column], errors="coerce")
        unit = DIST_UNIT.lower()

        if unit == "auto":
            unit = "au" if np.nanmedian(np.abs(values)) < 10 else "km"
            print(f"{column} column interpreted as {unit.upper()} "
                  f"(median |value| {np.nanmedian(np.abs(values)):.4g}); "
                  "set DIST_UNIT to override.")

        return values * 1.495978707e8 if unit == "au" else values

    df["dz_km"] = to_km("dz")
    df["dsat_km"] = to_km("dsat")

    dsat_from_column = not df["dsat_km"].isna().all()

    if not np.isfinite(
        df[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite CD or offset value found")

    # Identify the correct reference planet for each moon.
    df["parent"] = df["moon"].map(PARENT_BODY)

    unknown = df.loc[df["parent"].isna(), "moon"].unique()

    if len(unknown):
        raise ValueError(
            "No parent planet configured for: "
            + ", ".join(unknown)
            + ". Add each moon to PARENT_BODY."
        )

    if not dsat_from_column:
        if isinstance(PLANET_DISTANCE_AU, dict):
            df["dsat_km"] = df["parent"].map(PLANET_DISTANCE_AU) * 1.495978707e8
        elif PLANET_DISTANCE_AU is not None:
            df["dsat_km"] = PLANET_DISTANCE_AU * 1.495978707e8

        if df["dsat_km"].isna().all():
            print("Note: no dsat column or PLANET_DISTANCE_AU, so results "
                  "are in arcseconds only and dz is not used in the fit.")
        else:
            print("Using PLANET_DISTANCE_AU for the km conversion "
                  "(approximate: one distance for all dates).")

    return df


# ============================================================
# PREPARE GAUSSIAN CENTROID UNCERTAINTIES
# ============================================================

def prepare_centroid_errors(centres):
    """
    Prepare two tables:

    1. Moon uncertainties matched by date, image and moon.
    2. Parent-planet uncertainties matched by date and planet only.

    If several parent-planet image IDs exist on the same date,
    use the centroid fit with the smallest combined x/y uncertainty.
    """

    if centres.empty:
        raise ValueError("No successful Gaussian centroid fits")

    df = centres.copy()

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    df["moon"] = df["moon"].str.lower().str.strip()
    df["image"] = df["image"].str.lower().str.strip()
    df["axis"] = df["axis"].str.lower().str.strip()

    # Convert the x/y rows into one row per object and image.
    pivot = df.pivot_table(
        index=["date", "image", "moon"],
        columns="axis",
        values="centre_err_px",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None

    if not {"x", "y"}.issubset(pivot.columns):
        raise ValueError(
            "Centroid fits must have both x and y axes"
        )

    pivot = pivot.rename(columns={
        "x": "sigma_x",
        "y": "sigma_y",
    })

    # Moon-centroid errors retain their own image IDs.
    moon_errors = pivot.loc[
        ~pivot["moon"].isin(["saturn", "neptune"])
    ].copy()

    # Parent-planet errors do not require matching image IDs.
    parent_errors = pivot.loc[
        pivot["moon"].isin(["saturn", "neptune"])
    ].copy()

    parent_errors["combined_error"] = np.hypot(
        parent_errors["sigma_x"],
        parent_errors["sigma_y"],
    )

    # If several fits exist on a date, choose the best constrained
    # fit as a deterministic fallback.
    parent_errors = parent_errors.sort_values(
        ["date", "moon", "combined_error"],
        na_position="last",
    )

    counts = parent_errors.groupby(
        ["date", "moon"]
    ).size()

    if VERBOSE and (counts > 1).any():
        print(
            "Note: multiple parent-centroid image IDs exist "
            "on some dates; selecting the fit with the smallest "
            "combined x/y uncertainty."
        )

    parent_errors = parent_errors.drop_duplicates(
        subset=["date", "moon"],
        keep="first",
    )

    parent_errors = parent_errors.rename(columns={
        "moon": "parent",
        "sigma_x": "sigma_parent_x",
        "sigma_y": "sigma_parent_y",
    })

    return moon_errors, parent_errors


# ============================================================
# CALCULATE ANGULAR SEPARATIONS
# ============================================================

def calculate_angular_separations(centres):
    """Join the measurements to centroid errors and apply each CD matrix."""

    positions = load_relative_positions()

    moon_errors, parent_errors = prepare_centroid_errors(centres)

    # Match the observed moon's Gaussian errors.
    df = positions.merge(
        moon_errors[
            [
                "date", "image", "moon",
                "sigma_x", "sigma_y",
            ]
        ].rename(columns={
            "sigma_x": "sigma_moon_x",
            "sigma_y": "sigma_moon_y",
        }),
        on=["date", "image", "moon"],
        how="left",
        validate="many_to_one",
    )

    # Match Saturn or Neptune by DATE + PARENT BODY only.
    # The planet's image ID is deliberately ignored.
    df = df.merge(
        parent_errors[
            [
                "date", "parent",
                "sigma_parent_x", "sigma_parent_y",
            ]
        ],
        on=["date", "parent"],
        how="left",
        validate="many_to_one",
    )

    # Relative-coordinate uncertainties in pixels.
    # Moon and reference-planet centroid errors are assumed independent.
    df["sigma_dx"] = np.hypot(
        df["sigma_moon_x"], df["sigma_parent_x"]
    )

    df["sigma_dy"] = np.hypot(
        df["sigma_moon_y"], df["sigma_parent_y"]
    )

    # Observations whose centroid files are missing have no error. Rather
    # than dropping the weights for the whole moon (which makes the fit
    # overconfident), give them the typical error of the measured ones and
    # flag them.
    df["error_filled"] = False

    for column in ("sigma_dx", "sigma_dy"):
        finite = df[column][np.isfinite(df[column])]

        if len(finite):
            missing = ~np.isfinite(df[column])
            df.loc[missing, column] = float(finite.median())
            df.loc[missing, "error_filled"] = True

    dx = df["dx"].to_numpy(dtype=float)
    dy = df["dy"].to_numpy(dtype=float)

    # CD matrix for EACH row, converted from degrees to arcseconds.
    # [ CD1_1  CD1_2 ]
    # [ CD2_1  CD2_2 ]
    cd = df[
        ["cd1_1", "cd1_2", "cd2_1", "cd2_2"]
    ].to_numpy(dtype=float) * 3600.0

    # Tangent-plane angular offsets.
    sky_x = cd[:, 0] * dx + cd[:, 1] * dy
    sky_y = cd[:, 2] * dx + cd[:, 3] * dy

    separation = np.hypot(sky_x, sky_y)

    df["sky_x_arcsec"] = sky_x
    df["sky_y_arcsec"] = sky_y
    df["separation_arcsec"] = separation

    # Propagate independent x/y centroid errors through the CD matrix.
    sx = df["sigma_dx"].to_numpy(dtype=float)
    sy = df["sigma_dy"].to_numpy(dtype=float)

    valid = (
        (separation > 0)
        & np.isfinite(sx)
        & np.isfinite(sy)
        & (sx >= 0)
        & (sy >= 0)
    )

    errors = np.full(len(df), np.nan)

    # Derivatives of angular separation with respect to pixel dx, dy.
    grad_x = np.zeros(len(df))
    grad_y = np.zeros(len(df))

    grad_x[valid] = (
        sky_x[valid] * cd[valid, 0]
        + sky_y[valid] * cd[valid, 2]
    ) / separation[valid]

    grad_y[valid] = (
        sky_x[valid] * cd[valid, 1]
        + sky_y[valid] * cd[valid, 3]
    ) / separation[valid]

    errors[valid] = np.sqrt(
        (grad_x[valid] * sx[valid]) ** 2
        + (grad_y[valid] * sy[valid]) ** 2
    )

    df["error_arcsec"] = errors

    # Errors on sky_x and sky_y themselves (propagated through the CD matrix).
    df["sky_x_err_arcsec"] = np.sqrt(
        (cd[:, 0] * sx) ** 2 + (cd[:, 1] * sy) ** 2
    )
    df["sky_y_err_arcsec"] = np.sqrt(
        (cd[:, 2] * sx) ** 2 + (cd[:, 3] * sy) ** 2
    )

    # Make missing centroid matches visible.
    missing_moon = df["sigma_moon_x"].isna() | df["sigma_moon_y"].isna()
    missing_parent = (
        df["sigma_parent_x"].isna()
        | df["sigma_parent_y"].isna()
    )

    if missing_moon.any():
        print(
            f"Warning: {missing_moon.sum()} observation(s) have "
            "no matching moon centroid uncertainty, so they were given the "
            "typical error of the others (error_filled = True):"
        )
        print(
            "   " + ", ".join(
                f"{row.data} ({row.date})"
                for row in df.loc[missing_moon].itertuples()
            )
        )
        print(
            "   Check that profile files named <moon>_<image>_x/y exist in "
            "data/<date>/ for these."
        )

    if missing_parent.any():
        print(
            f"Warning: {missing_parent.sum()} observation(s) have "
            "no same-date reference-planet centroid uncertainty."
        )

        if VERBOSE:
            print(
                df.loc[
                    missing_parent,
                    ["data", "date", "parent"]
                ].to_string(index=False)
            )

    df = df.sort_values(["moon", "datetime"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUT_DIR / "angular_separations.csv"
    df.to_csv(output_path, index=False)

    print(f"Angular measurements saved to {output_path}")

    return df


# ============================================================
# ORBIT FIT: CENTRED ELLIPSE WITH A FREE PERIOD
# ============================================================
#
# Each moon's apparent path around its parent planet is an ellipse centred
# on the planet. For ANY orientation, shape and phase, such a path can be
# written
#
#     sky_x(t) = ax*sin(w t) + bx*cos(w t)
#     sky_y(t) = ay*sin(w t) + by*cos(w t)           w = 2*pi/P
#
# For a trial period P this is an ordinary linear least-squares problem, so
# the period is found by scanning a wide range of trial periods and keeping
# the best one. Nothing about the orbit (plane direction, period, size) is
# assumed in advance.

ARCSEC_PER_RAD = 206264.80624709636


def _basis(t, frequency):
    omega = 2.0 * np.pi * frequency
    return np.column_stack([np.sin(omega * t), np.cos(omega * t)])


def _weighted_fit(design, x, y, wx, wy, z=None, wz=None):
    """Weighted linear fit of x(t), y(t) (and z(t) if given) on the sin/cos basis.

    Returns (chi2, coefficients_x, coefficients_y, coefficients_z or None).
    """
    total = 0.0
    coefficients = []

    series = [(x, wx), (y, wy)]
    if z is not None:
        series.append((z, wz))

    for values, weights in series:
        root = np.sqrt(weights)
        coeff, *_ = np.linalg.lstsq(
            design * root[:, None], values * root, rcond=None
        )
        residual = (values - design @ coeff) * root
        total += float(np.sum(residual ** 2))
        coefficients.append(coeff)

    coeff_z = coefficients[2] if z is not None else None
    return total, coefficients[0], coefficients[1], coeff_z


def _axes_3d(coeff_x, coeff_y, coeff_z):
    """Semi-major and semi-minor axes of the 3D orbit ellipse."""
    matrix = np.array([coeff_x, coeff_y, coeff_z])
    singular = np.linalg.svd(matrix, compute_uv=False)
    return singular[0], singular[1]


def _ellipse_axes(coeff_x, coeff_y):
    """Semi-axes and orientation of the ellipse defined by the coefficients."""
    matrix = np.array([
        [coeff_x[0], coeff_x[1]],
        [coeff_y[0], coeff_y[1]],
    ])
    u, singular, _ = np.linalg.svd(matrix)
    major, minor = singular

    # Position angle of the major axis, measured from the +sky_y axis
    # towards the +sky_x axis (north through east, if sky_y is north and
    # sky_x is east), folded into 0-180 degrees.
    angle = np.degrees(np.arctan2(u[0, 0], u[1, 0])) % 180.0

    return major, minor, angle


def combine_into_visits(group):
    """
    Average images taken close together in time into one point per visit.

    Images a few seconds or minutes apart are not independent measurements
    of the orbit, so they are combined (weighted by their uncertainties when
    these are available) before fitting.
    """
    group = group.sort_values("datetime")

    gap_minutes = (
        group["datetime"].diff().dt.total_seconds().div(60.0).fillna(np.inf)
    )
    visit_id = (gap_minutes > VISIT_GAP_MINUTES).cumsum()

    rows = []

    for _, visit in group.groupby(visit_id):
        first = visit["datetime"].iloc[0]
        mean_time = first + (visit["datetime"] - first).mean()

        record = {"datetime": mean_time, "n_images": len(visit)}

        for axis in ("x", "y"):
            values = visit[f"sky_{axis}_arcsec"].to_numpy(dtype=float)
            errors = visit[f"sky_{axis}_err_arcsec"].to_numpy(dtype=float)

            if np.all(np.isfinite(errors)) and np.all(errors > 0):
                weights = 1.0 / errors ** 2
                record[axis] = float(np.sum(weights * values) / np.sum(weights))
                record[f"{axis}_err"] = float(1.0 / np.sqrt(np.sum(weights)))
            else:
                record[axis] = float(np.mean(values))
                record[f"{axis}_err"] = np.nan

        for column in ("dz_km", "dsat_km"):
            values = visit[column].to_numpy(dtype=float)
            record[column] = (
                float(values[np.isfinite(values)].mean())
                if np.isfinite(values).any() else np.nan
            )

        rows.append(record)

    return pd.DataFrame(rows)


def dz_consistency(visits, t, frequency):
    """
    Check the fitted period against the ephemeris depth offset dz.

    The period comes from the sky positions alone. A moon on a (nearly)
    circular orbit should also have dz varying as a sinusoid at that same
    period. This fits dz at the sky-derived period and reports how well it
    matches: a small ratio means the period is consistent with the
    ephemeris depth. dz comes from an ephemeris, so this is a check, not an
    independent measurement.
    """
    dz = visits["dz_km"].to_numpy(dtype=float)
    good = np.isfinite(dz)

    empty = {"dz_amplitude_km": np.nan, "dz_rms_km": np.nan,
             "dz_rms_over_amplitude": np.nan}

    if good.sum() < 4:
        return empty

    design = _basis(t[good], frequency)
    coeff, *_ = np.linalg.lstsq(design, dz[good], rcond=None)
    residual = dz[good] - design @ coeff
    amplitude = float(np.hypot(*coeff))
    rms = float(np.sqrt(np.mean(residual ** 2)))

    return {
        "dz_amplitude_km": amplitude,
        "dz_rms_km": rms,
        "dz_rms_over_amplitude": rms / amplitude if amplitude > 0 else np.nan,
    }


def fit_orbit_ellipse(t, x, y, sx, sy, z=None, sz=None):
    """
    Fit a centred ellipse with a free period.

    t         times in days
    x, y      sky offsets from the parent planet (arcsec)
    sx, sy    1-sigma uncertainties (arcsec); NaN if unknown
    z, sz     optional depth offset and its uncertainty, expressed in the
              same angular units as x and y (dz converted with the planet's
              distance). When given, the period is fitted to all three
              coordinates together.

    Returns a dictionary, or None when there are too few visits.
    """

    n = len(t)

    if n < (MIN_VISITS_WITH_DZ if z is not None else MIN_VISITS_FOR_FIT):
        return None

    span = float(np.ptp(t))

    if span <= 0:
        return None

    # 4 (or 6, with z) linear parameters + the period.
    n_coords = 3 if z is not None else 2
    n_linear = 2 * n_coords
    dof = n_coords * n - n_linear - 1

    if dof <= 0:
        return None

    weighted = (
        np.all(np.isfinite(sx)) and np.all(np.isfinite(sy))
        and np.all(sx > 0) and np.all(sy > 0)
    )

    wx = 1.0 / sx ** 2 if weighted else np.ones(n)
    wy = 1.0 / sy ** 2 if weighted else np.ones(n)

    if z is not None:
        wz = 1.0 / sz ** 2 if weighted else np.ones(n)
    else:
        wz = None

    # Scan evenly in frequency, which is how aliases are spaced.
    f_min = 1.0 / (MAX_PERIOD_SPAN_FACTOR * span)
    f_max = 1.0 / MIN_PERIOD_DAYS
    step = 1.0 / (PERIOD_SCAN_OVERSAMPLE * span)

    freqs = np.arange(f_min, f_max + step, step)
    chi2 = np.array([
        _weighted_fit(_basis(t, f), x, y, wx, wy, z, wz)[0] for f in freqs
    ])

    # Distinct local minima, best first.
    padded = np.r_[np.inf, chi2, np.inf]
    minima = [
        i for i in range(len(chi2))
        if padded[i + 1] < padded[i] and padded[i + 1] <= padded[i + 2]
    ]

    if not minima:
        return None

    minima.sort(key=lambda i: chi2[i])

    def refine(index):
        lower = freqs[max(index - 1, 0)]
        upper = freqs[min(index + 1, len(freqs) - 1)]
        result = minimize_scalar(
            lambda f: _weighted_fit(_basis(t, f), x, y, wx, wy, z, wz)[0],
            bounds=(lower, upper),
            method="bounded",
        )
        return float(result.fun), float(result.x), index

    # Refine each of the best dips before comparing them: a dip that looks
    # slightly worse on the grid can be exactly as good once refined.
    candidates = sorted(refine(index) for index in minima[:12])
    _, f_best, best = candidates[0]
    chi2_best, coeff_x, coeff_y, coeff_z = _weighted_fit(
        _basis(t, f_best), x, y, wx, wy, z, wz
    )

    # Scale: the residual variance per degree of freedom.
    # With trusted per-point errors the noise scale is at least 1, so a lucky
    # tiny chi-squared cannot make every alias look unlikely.
    scale = max(chi2_best / dof, 1.0 if weighted else 1e-12)

    # Period uncertainty: the range over which chi2 rises by one scale unit.
    left = best
    while left > 0 and chi2[left - 1] <= chi2_best + scale:
        left -= 1

    right = best
    while right < len(freqs) - 1 and chi2[right + 1] <= chi2_best + scale:
        right += 1

    period = 1.0 / f_best
    period_error = max(
        0.5 * (1.0 / freqs[left] - 1.0 / freqs[right]),
        0.5 * (1.0 / (f_best - step / 2) - 1.0 / (f_best + step / 2)),
    )

    # How distinct is the best period from its aliases?
    alternatives = []

    for chi2_c, f_c, _ in candidates[1:]:
        delta = (chi2_c - chi2_best) / scale

        if delta < ALIAS_DELTA_CHI2:
            alternatives.append((1.0 / f_c, delta))

    if len(candidates) > 1:
        next_period = 1.0 / candidates[1][1]
        next_delta = (candidates[1][0] - chi2_best) / scale
    else:
        next_period, next_delta = np.nan, np.inf

    at_limit = best in (0, len(freqs) - 1)

    if at_limit:
        status = "at edge of search range"
    elif alternatives:
        status = "ambiguous"
    else:
        status = "unique"

    # Ellipse size, shape and orientation.
    major, minor, angle = _ellipse_axes(coeff_x, coeff_y)

    # Uncertainty on the semi-axes by resampling at the best period.
    design = _basis(t, f_best)
    model_x = design @ coeff_x
    model_y = design @ coeff_y
    sigma_x = np.sqrt(scale / wx)
    sigma_y = np.sqrt(scale / wy)

    if z is not None:
        model_z = design @ coeff_z
        sigma_z = np.sqrt(scale / wz)
        major3d, minor3d = _axes_3d(coeff_x, coeff_y, coeff_z)
    else:
        model_z = sigma_z = None
        major3d = minor3d = np.nan

    rng = np.random.default_rng(12345)
    majors, minors, majors3d = [], [], []

    for _ in range(N_BOOTSTRAP):
        _, cx, cy, cz = _weighted_fit(
            design,
            model_x + rng.normal(0.0, sigma_x),
            model_y + rng.normal(0.0, sigma_y),
            wx,
            wy,
            None if z is None else model_z + rng.normal(0.0, sigma_z),
            wz,
        )
        a_sim, b_sim, _ = _ellipse_axes(cx, cy)
        majors.append(a_sim)
        minors.append(b_sim)

        if z is not None:
            majors3d.append(_axes_3d(cx, cy, cz)[0])

    squares = np.sum((x - model_x) ** 2) + np.sum((y - model_y) ** 2)
    if z is not None:
        squares += np.sum((z - model_z) ** 2)

    rms = float(np.sqrt(squares / (n_coords * n)))

    return {
        "weighted": bool(weighted),
        "period": period,
        "period_error": period_error,
        "status": status,
        "alternatives": alternatives,
        "next_period": next_period,
        "next_delta": next_delta,
        "major": major,
        "major_error": float(np.std(majors)),
        "used_dz": z is not None,
        "major3d": float(major3d),
        "minor3d": float(minor3d),
        "major3d_error": float(np.std(majors3d)) if majors3d else np.nan,
        "minor": minor,
        "minor_error": float(np.std(minors)),
        "angle": angle,
        "rms": rms,
        "chi2": chi2_best,
        "dof": dof,
        "coeff_x": coeff_x,
        "coeff_y": coeff_y,
        "frequency": f_best,
        "freqs": freqs,
        "scan_chi2": chi2,
        "scale": scale,
    }


def _plot_orbit(path, moon, year, origin, visits, t, result):
    """Sky-plane path, x(t) and y(t), and the period scan."""

    x = visits["x"].to_numpy(dtype=float)
    y = visits["y"].to_numpy(dtype=float)
    sx = visits["x_err"].to_numpy(dtype=float)
    sy = visits["y_err"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 2)
    ax_sky = fig.add_subplot(grid[0, 0])
    ax_scan = fig.add_subplot(grid[0, 1])
    ax_time = fig.add_subplot(grid[1, :])

    ax_sky.errorbar(
        x, y,
        xerr=np.where(np.isfinite(sx), sx, 0.0),
        yerr=np.where(np.isfinite(sy), sy, 0.0),
        fmt="o", capsize=2, label="visits",
    )
    ax_sky.plot(0, 0, "k+", markersize=12, label="parent planet")

    ax_time.errorbar(
        t, x,
        yerr=np.where(np.isfinite(sx), sx, 0.0),
        fmt="o", capsize=2, label="sky_x",
    )
    ax_time.errorbar(
        t, y,
        yerr=np.where(np.isfinite(sy), sy, 0.0),
        fmt="s", capsize=2, label="sky_y",
    )

    if result is not None:
        tau = np.linspace(0, 2 * np.pi, 400)
        ax_sky.plot(
            result["coeff_x"][0] * np.sin(tau) + result["coeff_x"][1] * np.cos(tau),
            result["coeff_y"][0] * np.sin(tau) + result["coeff_y"][1] * np.cos(tau),
            "-", label="fitted ellipse",
        )

        t_line = np.linspace(0, max(float(np.ptp(t)), result["period"]), 800)
        design = _basis(t_line, result["frequency"])
        ax_time.plot(t_line, design @ result["coeff_x"], "-", color="C0", alpha=0.7)
        ax_time.plot(t_line, design @ result["coeff_y"], "-", color="C1", alpha=0.7)

        periods = 1.0 / result["freqs"]
        ax_scan.semilogy(periods, result["scan_chi2"], "-")
        ax_scan.axhline(
            result["chi2"] + ALIAS_DELTA_CHI2 * result["scale"],
            color="gray", linestyle="--",
            label="aliases below this line are not ruled out",
        )
        ax_scan.axvline(result["period"], color="r", linestyle=":")
        ax_scan.set_xscale("log")
        ax_scan.set_xlabel("trial period (days)")
        ax_scan.set_ylabel("chi-squared (or residual sum of squares)")
        ax_scan.set_title(
            f"Period scan: best {result['period']:.3f} "
            f"+/- {result['period_error']:.3f} d ({result['status']})",
            fontsize=10,
        )
        ax_scan.legend(fontsize=7)
    else:
        ax_scan.axis("off")
        ax_sky.text(
            0.03, 0.97,
            f"Not fitted: fewer than {MIN_VISITS_FOR_FIT} separate visits.",
            transform=ax_sky.transAxes, va="top", fontsize=9,
        )

    ax_sky.set_xlabel("sky_x (arcsec)")
    ax_sky.set_ylabel("sky_y (arcsec)")
    ax_sky.set_aspect("equal", adjustable="datalim")
    ax_sky.grid(True, alpha=0.3)
    ax_sky.legend(fontsize=7)
    ax_sky.set_title("Path on the sky", fontsize=10)

    ax_time.set_xlabel(f"Days since {origin:%Y-%m-%d %H:%M}")
    ax_time.set_ylabel("Offset from parent planet (arcsec)")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=8)

    parent = PARENT_BODY.get(moon, "parent planet").capitalize()
    fig.suptitle(f"{moon.capitalize()} relative to {parent} ({year})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fit_orbits_by_year(df):
    """
    Fit every moon separately for every calendar year.

    Each year is processed completely independently: a year's fit uses only
    that year's observations, and its time axis starts at that year's first
    observation. A year with too few separate visits is reported as
    "not fitted" rather than being given another year's result.

    Outputs:
        results/angular_separations_<year>.csv
        results/orbit_fits_<year>.csv
        results/orbit_fits.csv                 (all years, with a year column)
        results/angular_plots/<year>/<moon>_orbit.png
    """

    ANGULAR_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["year"] = df["datetime"].dt.year

    columns = [
        "moon", "year", "n_images", "n_visits", "fit_status",
        "period_days", "period_error_days", "period_status",
        "next_best_period_days", "next_best_delta_chi2",
        "semi_major_arcsec", "semi_major_err_arcsec",
        "semi_minor_arcsec", "semi_minor_err_arcsec",
        "minor_over_major", "ring_opening_deg", "major_axis_pa_deg",
        "used_dz_in_fit", "semi_major_3d_arcsec",
        "semi_major_km", "semi_major_err_km",
        "rms_residual_arcsec", "weighted",
        "dz_amplitude_km", "dz_rms_km", "dz_rms_over_amplitude",
    ]

    all_records = []

    for year, year_data in df.groupby("year"):

        year_data.drop(columns=["year"]).to_csv(
            OUT_DIR / f"angular_separations_{year}.csv", index=False
        )
        print(f"Saved {OUT_DIR / f'angular_separations_{year}.csv'}")

        year_dir = ANGULAR_PLOT_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        year_records = []

        for moon, moon_year in year_data.groupby("moon"):
            usable = moon_year.loc[
                np.isfinite(moon_year["sky_x_arcsec"])
                & np.isfinite(moon_year["sky_y_arcsec"])
            ]

            if usable.empty:
                continue

            visits = combine_into_visits(usable)
            origin = visits["datetime"].min()
            t = (visits["datetime"] - origin).dt.total_seconds().to_numpy() / 86400.0

            x_v = visits["x"].to_numpy(dtype=float)
            y_v = visits["y"].to_numpy(dtype=float)
            sx_v = visits["x_err"].to_numpy(dtype=float)
            sy_v = visits["y_err"].to_numpy(dtype=float)

            # Depth offset as a third coordinate, in the same angular units
            # as x and y: dz / (planet distance per arcsecond).
            z_v = sz_v = None

            if USE_DZ_IN_FIT:
                dz_v = visits["dz_km"].to_numpy(dtype=float)
                d_v = visits["dsat_km"].to_numpy(dtype=float)

                if np.all(np.isfinite(dz_v)) and np.all(np.isfinite(d_v)):
                    z_v = dz_v / (d_v / ARCSEC_PER_RAD)
                    sz_v = np.hypot(sx_v, sy_v) / np.sqrt(2.0)
                else:
                    print(
                        f"Note: {moon} {year}: dz or dSat missing, "
                        "fitting sky positions only."
                    )

            result = fit_orbit_ellipse(t, x_v, y_v, sx_v, sy_v, z_v, sz_v)

            record = {
                "moon": moon,
                "year": year,
                "n_images": int(visits["n_images"].sum()),
                "n_visits": len(visits),
            }

            if result is None:
                needed = MIN_VISITS_WITH_DZ if z_v is not None else MIN_VISITS_FOR_FIT
                record["fit_status"] = (
                    f"not fitted: needs at least {needed} separate visits"
                )
                print(
                    f"{moon} {year}: not fitted "
                    f"({len(visits)} visit(s); need {needed})."
                )
            else:
                finite_d = visits["dsat_km"].to_numpy(dtype=float)
                finite_d = finite_d[np.isfinite(finite_d)]
                distance = float(finite_d.mean()) if len(finite_d) else np.nan
                km_per_arcsec = distance / ARCSEC_PER_RAD

                record.update({
                    "fit_status": "fitted",
                    "period_days": result["period"],
                    "period_error_days": result["period_error"],
                    "period_status": result["status"],
                    "next_best_period_days": result["next_period"],
                    "next_best_delta_chi2": result["next_delta"],
                    "semi_major_arcsec": result["major"],
                    "semi_major_err_arcsec": result["major_error"],
                    "semi_minor_arcsec": result["minor"],
                    "semi_minor_err_arcsec": result["minor_error"],
                    "minor_over_major": result["minor"] / result["major"],
                    "ring_opening_deg": float(np.degrees(np.arcsin(
                        min(result["minor"] / result["major"], 1.0)
                    ))),
                    "major_axis_pa_deg": result["angle"],
                    "used_dz_in_fit": result["used_dz"],
                    "semi_major_3d_arcsec": result["major3d"],
                    # With dz the 3D orbit radius is used; without it, the
                    # sky ellipse's semi-major axis.
                    "semi_major_km": (
                        (result["major3d"] if result["used_dz"]
                         else result["major"]) * km_per_arcsec
                        if np.isfinite(distance) else np.nan
                    ),
                    "semi_major_err_km": (
                        (result["major3d_error"] if result["used_dz"]
                         else result["major_error"]) * km_per_arcsec
                        if np.isfinite(distance) else np.nan
                    ),
                    "rms_residual_arcsec": result["rms"],
                    "weighted": result["weighted"],
                })
                record.update(dz_consistency(visits, t, result["frequency"]))

                print(
                    f"{moon} {year}: period = {result['period']:.3f} "
                    f"+/- {result['period_error']:.3f} d "
                    f"({result['status']}"
                    f"{', with dz' if result['used_dz'] else ', sky only'}); "
                    f"semi-major axis = {result['major']:.1f} arcsec"
                )

                if result["status"] == "ambiguous":
                    others = ", ".join(
                        f"{p:.3f} d" for p, _ in result["alternatives"][:3]
                    )
                    print(
                        f"    Warning: other periods fit nearly as well "
                        f"({others}). More visits at different spacings "
                        f"are needed to separate them."
                    )

                if not result["weighted"]:
                    print(
                        f"    Note: {moon} {year} was fitted without "
                        "per-point uncertainties (some are missing)."
                    )

            year_records.append(record)

            _plot_orbit(
                year_dir / f"{moon}_orbit.png",
                moon, year, origin, visits, t, result,
            )

        year_table = pd.DataFrame(year_records).reindex(columns=columns)
        year_table.to_csv(OUT_DIR / f"orbit_fits_{year}.csv", index=False)

        all_records.extend(year_records)

    pd.DataFrame(all_records).reindex(columns=columns).to_csv(
        OUT_DIR / "orbit_fits.csv", index=False
    )

    print(f"Orbit plots saved in {ANGULAR_PLOT_DIR}")
    print(f"Fit summary saved to {OUT_DIR / 'orbit_fits.csv'}")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    centres = fit_all_profiles()

    if centres.empty:
        print("No successful Gaussian fits. Stopping.")
        return

    angular = calculate_angular_separations(centres)

    if USE_ORBIT_FIT_MODULE:
        import orbit_fit

        orbit_fit.fit_and_plot_orbits(angular, OUT_DIR, None, PARENT_BODY)
    else:
        fit_orbits_by_year(angular)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()