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
    Data, Date, Time, CD1_1, CD1_2, CD2_1, CD2_2, dX, dY

Outputs
-------
results/centres.csv
results/angular_separations.csv
results/angular_separations_<year>.csv      one file per observing year
results/sinusoidal_fits_<year>.csv          one file per observing year
results/sinusoidal_fits.csv                 summary of the above (has a year column)
results/plots/
results/angular_plots/<year>/

Each observing year is fitted completely on its own: data from different
years are never combined in a sinusoidal fit.

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
from scipy.optimize import curve_fit

import orbit_fit

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

# Approximate orbital periods, in days, used as fit starting values.
# The fit is constrained to +/- 50% of the listed value by default.
INITIAL_PERIODS = {
    "titan": 15.945,
    "rhea": 4.518,
    "tethys": 1.888,
    "dione": 2.737,
    "enceladus": 1.370,
    "triton": 5.877,
}

USE_PERIOD_PRIOR = True

# A yearly fit is attempted when a moon has at least this many observations
# in that year.
MIN_OBSERVATIONS_FOR_FIT = 6

# Images taken minutes apart on the same night are not independent
# measurements of the orbit. A warning is printed when a yearly fit rests on
# fewer distinct observing nights than this.
MIN_NIGHTS_FOR_RELIABLE_FIT = 5

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

    # Make missing centroid matches visible.
    missing_moon = df["sigma_moon_x"].isna() | df["sigma_moon_y"].isna()
    missing_parent = (
        df["sigma_parent_x"].isna()
        | df["sigma_parent_y"].isna()
    )

    if missing_moon.any():
        print(
            f"Warning: {missing_moon.sum()} observation(s) have "
            "no matching moon centroid uncertainty."
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
# SINUSOIDAL FIT
# ============================================================
def sinusoid(t, offset, amplitude, period, phase):
    return offset + amplitude * np.abs(np.sin(2 * np.pi * t / period + phase))

def fit_sinusoid(t, y, errors, moon, label=None):
    """
    Fit an empirical sinusoid.

    Returns:
        (t0, parameters, covariance), or None if the data
        do not support a reliable fit.

    At least MIN_OBSERVATIONS_FOR_FIT observations spanning a non-zero
    time interval are required. This is a practical minimum, not a
    guarantee that the fitted period is well constrained.

    `label` is only used in printed messages (for example "titan 2026").
    """

    label = label or moon

    good = np.isfinite(t) & np.isfinite(y)
    t = np.asarray(t)[good]
    y = np.asarray(y)[good]
    errors = np.asarray(errors)[good]

    if len(y) < MIN_OBSERVATIONS_FOR_FIT or np.ptp(t) <= 0:
        return None

    t0 = t.min()
    t = t - t0
    span = np.ptp(t)

    period_guess = INITIAL_PERIODS.get(moon, span / 2)

    if span < period_guess:
        print(
            f"Warning: {label}: the observations span only "
            f"{span:.2f} days, shorter than the starting period "
            f"of {period_guess:.3f} days. The fit may be uncertain."
        )

    amplitude_guess = max(
        (np.max(y) - np.min(y)) / 2,
        1e-5,
    )
    offset_guess = np.mean(y)

    if USE_PERIOD_PRIOR:
        minimum_period = max(0.05, period_guess * 0.5)
        maximum_period = period_guess * 1.5
    else:
        minimum_period = max(0.05, span / 10)
        maximum_period = max(minimum_period * 1.01, span * 2)

    weighted = (
        len(errors) == len(y)
        and np.all(np.isfinite(errors))
        and np.all(errors > 0)
    )

    fit_options = {}

    if weighted:
        fit_options["sigma"] = errors
        fit_options["absolute_sigma"] = True

    try:
        parameters, covariance = curve_fit(
            sinusoid,
            t,
            y,
            p0=[
                offset_guess,
                amplitude_guess,
                period_guess,
                0.0,
            ],
            bounds=(
                [
                    -np.inf, 0,
                    minimum_period, -2 * np.pi,
                ],
                [
                    np.inf, np.inf,
                    maximum_period, 2 * np.pi,
                ],
            ),
            maxfev=50000,
            **fit_options,
        )

        return t0, parameters, covariance

    except (RuntimeError, ValueError) as error:
        print(f"Sinusoidal fit failed for {label}: {error}")
        return None


def plot_angular_separations(df):
    """
    Fit and plot every moon separately for every calendar year.

    Each year is processed completely independently: the sinusoidal fit
    for a year uses only that year's observations, and its time axis
    starts at that year's first observation. Data from different years
    are never combined, and a year with too few observations is reported
    as "not fitted" rather than being given another year's fit.

    Outputs:
        results/angular_separations_<year>.csv
        results/sinusoidal_fits_<year>.csv
        results/sinusoidal_fits.csv            (all years, with a year column)
        results/angular_plots/<year>/<moon>_separation.png

    New years are handled automatically.
    """

    ANGULAR_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["year"] = df["datetime"].dt.year

    fit_columns = [
        "moon", "year", "n_observations", "n_nights", "fit_type",
        "offset_arcsec", "amplitude_arcsec",
        "period_days", "period_error_days", "phase_rad",
    ]

    fit_records = []

    for year, year_data in df.groupby("year"):

        # Save this year's measurements.
        csv_path = OUT_DIR / f"angular_separations_{year}.csv"

        year_data.drop(columns=["year"]).to_csv(
            csv_path,
            index=False,
        )

        print(f"Saved {csv_path}")

        year_dir = ANGULAR_PLOT_DIR / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        year_records = []

        for moon, moon_year in year_data.groupby("moon"):
            group = moon_year.sort_values("datetime").copy()

            group = group.loc[
                np.isfinite(group["separation_arcsec"])
            ]

            if group.empty:
                continue

            year_origin = group["datetime"].min()

            t_year = (
                group["datetime"] - year_origin
            ).dt.total_seconds().to_numpy() / 86400.0

            y_year = group["separation_arcsec"].to_numpy(dtype=float)
            error_year = group["error_arcsec"].to_numpy(dtype=float)

            n_nights = int(group["datetime"].dt.normalize().nunique())

            have_error = (
                np.isfinite(error_year) & (error_year >= 0)
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            # Observations with valid error bars.
            ax.errorbar(
                t_year[have_error],
                y_year[have_error],
                yerr=error_year[have_error],
                fmt="o",
                capsize=3,
                label="Observed separation with uncertainty",
            )

            # Retain observations even when their errors are missing.
            ax.plot(
                t_year[~have_error],
                y_year[~have_error],
                "o",
                label="Observed separation without uncertainty",
            )

            # Fit this year on its own. There is deliberately no fallback
            # to a fit made from other years' data.
            yearly_fit = None

            if len(group) >= MIN_OBSERVATIONS_FOR_FIT:
                yearly_fit = fit_sinusoid(
                    t_year,
                    y_year,
                    error_year,
                    moon,
                    label=f"{moon} {year}",
                )

                if (
                    yearly_fit is not None
                    and n_nights < MIN_NIGHTS_FOR_RELIABLE_FIT
                ):
                    print(
                        f"Warning: {moon} {year}: the fit uses "
                        f"{len(group)} images but only {n_nights} "
                        f"distinct night(s); the period is likely to be "
                        f"poorly constrained."
                    )

            if yearly_fit is not None:
                t0_fit, parameters, covariance = yearly_fit

                line_t = np.linspace(
                    t_year.min(),
                    t_year.max(),
                    800,
                )
                line_y = sinusoid(line_t - t0_fit, *parameters)

                ax.plot(
                    line_t,
                    line_y,
                    "-",
                    label=(
                        f"{year} sinusoidal fit "
                        f"(P = {parameters[2]:.3f} days)"
                    ),
                )

                if np.all(np.isfinite(covariance)):
                    parameter_errors = np.sqrt(
                        np.maximum(np.diag(covariance), 0)
                    )
                    period_error = parameter_errors[2]
                else:
                    period_error = np.nan

                year_records.append({
                    "moon": moon,
                    "year": year,
                    "n_observations": len(group),
                    "n_nights": n_nights,
                    "fit_type": "yearly fit",
                    "offset_arcsec": parameters[0],
                    "amplitude_arcsec": parameters[1],
                    "period_days": parameters[2],
                    "period_error_days": period_error,
                    "phase_rad": parameters[3],
                })

                print(
                    f"{moon} {year}: "
                    f"period = {parameters[2]:.4f} days"
                )

            else:
                # Explicitly show why the fit is absent.
                ax.text(
                    0.03,
                    0.97,
                    "No sinusoidal fit for this year: insufficient "
                    "observations.\n"
                    f"At least {MIN_OBSERVATIONS_FOR_FIT} observations "
                    "are required.",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                )

                year_records.append({
                    "moon": moon,
                    "year": year,
                    "n_observations": len(group),
                    "n_nights": n_nights,
                    "fit_type": "not fitted: insufficient data",
                    "offset_arcsec": np.nan,
                    "amplitude_arcsec": np.nan,
                    "period_days": np.nan,
                    "period_error_days": np.nan,
                    "phase_rad": np.nan,
                })

                print(
                    f"{moon} {year}: no fit; "
                    f"{len(group)} observations available."
                )

            parent = PARENT_BODY.get(moon, "parent planet")

            ax.set_title(
                f"{moon.capitalize()}: angular separation "
                f"from {parent.capitalize()} ({year})"
            )
            ax.set_xlabel(
                f"Days since {year_origin:%Y-%m-%d %H:%M}"
            )
            ax.set_ylabel("Angular separation (arcseconds)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()

            fig.savefig(
                year_dir / f"{moon}_separation.png",
                dpi=180,
            )
            plt.close(fig)

        # One fit-summary file per year.
        pd.DataFrame(
            year_records,
            columns=fit_columns,
        ).to_csv(
            OUT_DIR / f"sinusoidal_fits_{year}.csv",
            index=False,
        )

        fit_records.extend(year_records)

    # Summary of all years. Each row is still an independent yearly fit.
    pd.DataFrame(
        fit_records,
        columns=fit_columns,
    ).to_csv(
        OUT_DIR / "sinusoidal_fits.csv",
        index=False,
    )

    print(f"Yearly plots saved in {ANGULAR_PLOT_DIR}")
    print(f"Fit summary saved to {OUT_DIR / 'sinusoidal_fits.csv'}")

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

    plot_angular_separations(angular)

    orbit_fit.fit_and_plot_orbits(
        angular, OUT_DIR, INITIAL_PERIODS, PARENT_BODY
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

