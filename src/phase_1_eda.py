"""
Phase 1: Data Acquisition & Exploratory Data Analysis
Kepler Exoplanet Classification -- Explainable ML System

Downloads the NASA Exoplanet Archive cumulative KOI table,
performs thorough EDA, and saves all outputs to disk.

No Phase 2+ operations (no splits, scaling, modelling, or SMOTE).

Compatibility: Python 3.12, pandas 2.x, NumPy 2.x, Matplotlib 3.9+,
               seaborn 0.13+, scipy 1.x, Windows PowerShell (ASCII-safe output).
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Stdout: force UTF-8 on Windows so the Markdown report write never fails,
# but keep terminal output ASCII-only so CP1252 consoles do not choke.
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # reconfigure not available in all contexts -- safe to skip

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
REPORTS_P1      = ROOT / "reports" / "phase_1"
PLOTS_P1        = ROOT / "plots" / "phase_1"
RAW_CSV         = DATA_RAW / "cumulative_kois.csv"

for _d in [DATA_RAW, DATA_PROCESSED, REPORTS_P1, PLOTS_P1]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NASA_URLS = [
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv",
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv",
]

LEAKAGE_COLUMNS = [
    "koi_score",
    "koi_pdisposition",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]

FEATURE_COLUMNS = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_impact",
    "koi_model_snr",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
]

DIST_FEATURES   = ["koi_period", "koi_depth", "koi_prad", "koi_duration"]
QUANTILE_LEVELS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


# ===========================================================================
# 1. DATA ACQUISITION
# ===========================================================================
def download_dataset() -> None:
    """Download cumulative KOI CSV from NASA Exoplanet Archive if not present."""
    if RAW_CSV.exists() and RAW_CSV.stat().st_size > 10_000:
        size_mb = RAW_CSV.stat().st_size / 1_048_576
        print(f"[INFO] Raw dataset already present: {RAW_CSV} ({size_mb:.1f} MB) -- skipping download.")
        return

    print("[INFO] Downloading cumulative KOI table from NASA Exoplanet Archive...")
    for url in NASA_URLS:
        try:
            print(f"  Trying: {url[:80]}...")
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            total = 0
            with open(RAW_CSV, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65_536):
                    fh.write(chunk)
                    total += len(chunk)
            size_mb = total / 1_048_576
            print(f"  [OK] Downloaded {size_mb:.2f} MB -> {RAW_CSV}")
            return
        except Exception as exc:
            print(f"  [FAIL] {exc} -- trying next URL...")

    print("\n" + "=" * 70)
    print("CRITICAL: Could not download the NASA dataset automatically.")
    print("Please download it manually:")
    print("  URL: https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
          "query=select+*+from+cumulative&format=csv")
    print(f"  Save as: {RAW_CSV}")
    print("Then re-run this script.")
    print("=" * 70)
    sys.exit(1)


# ===========================================================================
# 2. LOAD DATA
# ===========================================================================
def load_data() -> pd.DataFrame:
    """Load raw CSV. Handles comment rows that NASA sometimes prepends."""
    print(f"\n[INFO] Loading {RAW_CSV} ...")

    # Read, skipping comment lines that start with '#'
    df = pd.read_csv(RAW_CSV, comment="#", low_memory=False)

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ===========================================================================
# 3. AUDIT DATASET
# ===========================================================================
def audit_dataset(df: pd.DataFrame) -> None:
    """Print comprehensive dataset audit to console."""
    print("\n" + "=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)
    print(f"Rows       : {df.shape[0]:,}")
    print(f"Columns    : {df.shape[1]}")

    print("\n--- Column names ---")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")

    print("\n--- Data types ---")
    print(df.dtypes.to_string())

    print("\n--- First 5 rows ---")
    print(df.head(5).to_string())

    print("\n--- Random sample (5 rows) ---")
    print(df.sample(5, random_state=RANDOM_SEED).to_string())

    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows : {dup_count:,}")

    total_missing = df.isnull().sum().sum()
    print(f"Total missing  : {total_missing:,}")

    print("\n--- koi_disposition unique values ---")
    if "koi_disposition" in df.columns:
        vc = df["koi_disposition"].value_counts(dropna=False)
        print(vc.to_string())
    else:
        print("  WARNING: koi_disposition column not found!")


# ===========================================================================
# 4. CREATE TARGET DATASETS
# ===========================================================================
def create_target_datasets(df: pd.DataFrame):
    """
    Split into labeled (CONFIRMED / FALSE POSITIVE) and candidate sets.
    Returns (labeled_df, candidate_df).
    """
    print("\n[INFO] Creating target datasets...")

    labeled_df = df[
        df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    ].copy()

    candidate_df = df[
        df["koi_disposition"] == "CANDIDATE"
    ].copy()

    labeled_df["target"] = (
        labeled_df["koi_disposition"] == "CONFIRMED"
    ).astype(int)

    # Verify target contains only 0 and 1
    unique_targets = sorted(labeled_df["target"].unique().tolist())
    assert unique_targets == [0, 1], (
        f"Unexpected target values: {unique_targets}"
    )

    confirmed_count = int((labeled_df["target"] == 1).sum())
    fp_count        = int((labeled_df["target"] == 0).sum())
    candidate_count = len(candidate_df)
    labeled_count   = len(labeled_df)

    print(f"  CONFIRMED     : {confirmed_count:,}")
    print(f"  FALSE POSITIVE: {fp_count:,}")
    print(f"  CANDIDATE     : {candidate_count:,}")
    print(f"  Labeled total : {labeled_count:,}")

    return labeled_df, candidate_df


# ===========================================================================
# 5. REMOVE LEAKAGE COLUMNS
# ===========================================================================
def remove_leakage_columns(labeled_df: pd.DataFrame,
                            candidate_df: pd.DataFrame):
    """Drop columns that would leak disposition information."""
    print("\n[INFO] Removing leakage columns...")
    present_leakage = [c for c in LEAKAGE_COLUMNS if c in labeled_df.columns]
    missing_leakage = [c for c in LEAKAGE_COLUMNS if c not in labeled_df.columns]

    labeled_df   = labeled_df.drop(columns=present_leakage, errors="ignore")
    candidate_df = candidate_df.drop(columns=present_leakage, errors="ignore")

    print(f"  Removed: {present_leakage}")
    if missing_leakage:
        print(f"  Already absent: {missing_leakage}")

    # Verify none remain
    remaining = [c for c in LEAKAGE_COLUMNS if c in labeled_df.columns]
    assert len(remaining) == 0, f"Leakage columns still present: {remaining}"
    print("  [OK] Verified: no leakage columns in labeled set")

    return labeled_df, candidate_df


# ===========================================================================
# 6. VALIDATE FEATURES
# ===========================================================================
def validate_features(labeled_df: pd.DataFrame) -> None:
    """Stop if any required feature column is missing."""
    print("\n[INFO] Validating feature columns...")
    missing = [f for f in FEATURE_COLUMNS if f not in labeled_df.columns]
    if missing:
        print(f"\nCRITICAL: Missing feature columns: {missing}")
        print("STOP -- cannot continue without required features.")
        sys.exit(1)
    print(f"  [OK] All {len(FEATURE_COLUMNS)} feature columns present")


# ===========================================================================
# 7. CLASS BALANCE
# ===========================================================================
def analyze_class_balance(labeled_df: pd.DataFrame) -> dict:
    """Calculate and save class balance report."""
    print("\n[INFO] Analysing class balance...")

    vc    = labeled_df["target"].value_counts()
    total = len(labeled_df)

    confirmed_n   = int(vc.get(1, 0))
    fp_n          = int(vc.get(0, 0))
    confirmed_pct = confirmed_n / total * 100
    fp_pct        = fp_n / total * 100

    balance_df = pd.DataFrame({
        "class":      ["CONFIRMED", "FALSE_POSITIVE"],
        "target":     [1, 0],
        "count":      [confirmed_n, fp_n],
        "percentage": [round(confirmed_pct, 2), round(fp_pct, 2)],
    })
    balance_df.to_csv(REPORTS_P1 / "class_balance.csv", index=False)

    print(f"  CONFIRMED     : {confirmed_n:,}  ({confirmed_pct:.1f}%)")
    print(f"  FALSE POSITIVE: {fp_n:,}  ({fp_pct:.1f}%)")
    imbalanced = abs(confirmed_pct - fp_pct) > 15
    print(f"  Imbalanced    : {'YES' if imbalanced else 'NO'} (>15% difference threshold)")

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["CONFIRMED\n(target=1)", "FALSE POSITIVE\n(target=0)"],
        [confirmed_n, fp_n],
        color=["#2196F3", "#F44336"],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, pct in zip(bars, [confirmed_pct, fp_pct]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Class Balance: CONFIRMED vs FALSE POSITIVE", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(PLOTS_P1 / "class_balance.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_P1 / 'class_balance.png'}")

    return {
        "confirmed_n":   confirmed_n,
        "fp_n":          fp_n,
        "confirmed_pct": confirmed_pct,
        "fp_pct":        fp_pct,
        "imbalanced":    imbalanced,
    }


# ===========================================================================
# 8. MISSINGNESS
# ===========================================================================
def analyze_missingness(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and save missingness for the 11 feature columns."""
    print("\n[INFO] Analysing missingness...")

    features_df = labeled_df[FEATURE_COLUMNS]
    miss_count  = features_df.isnull().sum()
    miss_pct    = miss_count / len(labeled_df) * 100

    miss_df = pd.DataFrame({
        "feature":       FEATURE_COLUMNS,
        "missing_count": miss_count.values,
        "missing_pct":   miss_pct.round(2).values,
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    miss_df.to_csv(REPORTS_P1 / "missingness.csv", index=False)

    print("  Missingness (descending):")
    for _, row in miss_df.iterrows():
        flag = "  [>20% MISSING]" if row["missing_pct"] > 20 else ""
        print(f"    {row['feature']:<18}: {row['missing_count']:>5,}  ({row['missing_pct']:.1f}%){flag}")

    high_miss = miss_df[miss_df["missing_pct"] > 20]["feature"].tolist()
    if high_miss:
        print(f"  Features >20% missing: {high_miss}")
    else:
        print("  No features exceed 20% missing.")

    return miss_df


# ===========================================================================
# 9. SUMMARY STATISTICS
# ===========================================================================
def generate_summary_statistics(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and save descriptive statistics including skewness."""
    print("\n[INFO] Generating summary statistics...")

    feats     = labeled_df[FEATURE_COLUMNS]
    desc      = feats.describe().T      # rows = features, cols = count/mean/std/...
    skew_vals = feats.skew()
    desc["skewness"] = skew_vals

    # pandas 2.x: describe() produces '50%' -- rename to 'median' for clarity
    desc = desc.rename(columns={"50%": "median"})
    desc.to_csv(REPORTS_P1 / "summary_statistics.csv")
    print(f"  Saved: {REPORTS_P1 / 'summary_statistics.csv'}")
    return desc


# ===========================================================================
# 10. SKEWNESS
# ===========================================================================
def analyze_skewness(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Identify and report heavily skewed features."""
    print("\n[INFO] Analysing skewness...")

    # pandas 2.x: .skew() returns a Series indexed by column names.
    # Build the DataFrame explicitly to avoid reset_index naming differences
    # across pandas minor versions (reset_index(names=...) added in 1.5 but
    # the value-column name varies; explicit construction is always safe).
    skew_series = labeled_df[FEATURE_COLUMNS].skew()
    skew_df = pd.DataFrame({
        "feature":  skew_series.index.tolist(),
        "skewness": skew_series.values,
    })

    skew_df["abs_skew"] = skew_df["skewness"].abs()
    skew_df = skew_df.sort_values("abs_skew", ascending=False).reset_index(drop=True)

    print("  Skewness values (|skew| descending):")
    for _, row in skew_df.iterrows():
        tag = ""
        if row["abs_skew"] > 10:
            tag = "  [EXTREME]"
        elif row["abs_skew"] > 1:
            tag = "  [heavy]"
        print(f"    {row['feature']:<18}: {row['skewness']:>8.2f}{tag}")

    return skew_df


# ===========================================================================
# 11. DISTRIBUTION PLOTS
# ===========================================================================
def generate_distribution_plots(labeled_df: pd.DataFrame) -> None:
    """KDE / histogram comparing classes for 4 key features."""
    print("\n[INFO] Generating distribution plots...")

    label_map = {0: "FALSE POSITIVE", 1: "CONFIRMED"}
    colors    = {0: "#F44336", 1: "#2196F3"}

    for feat in DIST_FEATURES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for cls in [0, 1]:
            subset = labeled_df.loc[labeled_df["target"] == cls, feat].dropna()

            # Histogram (density-normalised)
            axes[0].hist(
                subset,
                bins=60,
                alpha=0.5,
                label=label_map[cls],
                color=colors[cls],
                density=True,
            )

            # KDE via scipy -- avoids pandas 2.x FutureWarning from Series.plot.kde
            if len(subset) > 1:
                kde = stats.gaussian_kde(subset)
                x_min = float(subset.min())
                x_max = float(subset.max())
                x_grid = np.linspace(x_min, x_max, 300)
                axes[1].plot(x_grid, kde(x_grid),
                             label=label_map[cls], color=colors[cls])

        axes[0].set_title(f"{feat} -- Histogram (density)")
        axes[0].set_xlabel(feat)
        axes[0].set_ylabel("Density")
        axes[0].legend()

        axes[1].set_title(f"{feat} -- KDE")
        axes[1].set_xlabel(feat)
        axes[1].set_ylabel("Density")
        axes[1].legend()

        plt.suptitle(f"Distribution: {feat}", fontsize=13, y=1.02)
        plt.tight_layout()

        out = PLOTS_P1 / f"dist_{feat}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ===========================================================================
# 12. BOXPLOTS
# ===========================================================================
def generate_boxplots(labeled_df: pd.DataFrame) -> None:
    """Class-separated boxplots for 4 key features."""
    print("\n[INFO] Generating boxplots...")

    palette = {0: "#F44336", 1: "#2196F3"}

    for feat in DIST_FEATURES:
        fig, ax = plt.subplots(figsize=(7, 5))

        data_to_plot = [
            labeled_df.loc[labeled_df["target"] == cls, feat].dropna().values
            for cls in [0, 1]
        ]

        # FIX: Matplotlib 3.9+ removed the 'labels' kwarg from boxplot().
        # Use 'tick_labels' instead (introduced in 3.9, replacing 'labels').
        bp = ax.boxplot(
            data_to_plot,
            tick_labels=["FALSE POSITIVE", "CONFIRMED"],
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, cls in zip(bp["boxes"], [0, 1]):
            patch.set_facecolor(palette[cls])
            patch.set_alpha(0.7)

        ax.set_title(f"Boxplot: {feat} by Class", fontsize=12)
        ax.set_ylabel(feat)
        plt.tight_layout()

        out = PLOTS_P1 / f"boxplot_{feat}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")


# ===========================================================================
# 13. CORRELATION HEATMAP
# ===========================================================================
def generate_correlation_heatmap(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for all 11 features."""
    print("\n[INFO] Generating correlation heatmap...")

    corr = labeled_df[FEATURE_COLUMNS].corr(method="pearson")
    corr.to_csv(REPORTS_P1 / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    # np.ones with shape tuple -- avoids potential numpy 2.x issue with ones_like on DataFrame
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson Correlation -- 11 KOI Features", fontsize=13)
    plt.tight_layout()
    out = PLOTS_P1 / "correlation_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # Find notable pairs (|r| > 0.5, excluding self-correlation)
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                pairs.append((corr.columns[i], corr.columns[j], round(r, 3)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("  Notable correlated pairs (|r| > 0.5):")
    if pairs:
        for a, b, r in pairs:
            print(f"    {a}  <->  {b} :  r = {r}")
    else:
        print("    None found.")

    return corr


# ===========================================================================
# 14. QUANTILE ANALYSIS
# ===========================================================================
def analyze_quantiles(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Compute quantiles for all 11 features."""
    print("\n[INFO] Computing feature quantiles...")

    q_df = labeled_df[FEATURE_COLUMNS].quantile(QUANTILE_LEVELS)
    q_df.index = [f"{int(q * 100)}%" for q in QUANTILE_LEVELS]
    q_df.T.to_csv(REPORTS_P1 / "feature_quantiles.csv")
    print(f"  Saved: {REPORTS_P1 / 'feature_quantiles.csv'}")
    return q_df


# ===========================================================================
# 15. CLASS SEPARABILITY
# ===========================================================================
def analyze_class_separability(labeled_df: pd.DataFrame) -> dict:
    """
    Compare CONFIRMED vs FALSE POSITIVE for 4 key features.
    Reports median, std, and Mann-Whitney U test.
    Returns observations dict.
    """
    print("\n[INFO] Analysing class separability...")
    obs = {}
    for feat in DIST_FEATURES:
        confirmed = labeled_df.loc[labeled_df["target"] == 1, feat].dropna()
        fp        = labeled_df.loc[labeled_df["target"] == 0, feat].dropna()

        med_conf = confirmed.median()
        med_fp   = fp.median()
        std_conf = confirmed.std()
        std_fp   = fp.std()

        try:
            _, p_val = stats.mannwhitneyu(confirmed, fp, alternative="two-sided")
        except Exception:
            p_val = np.nan

        p_str = f"{p_val:.6f}" if not np.isnan(p_val) else "n/a"

        obs[feat] = {
            "median_confirmed": round(float(med_conf), 4),
            "median_fp":        round(float(med_fp), 4),
            "std_confirmed":    round(float(std_conf), 4),
            "std_fp":           round(float(std_fp), 4),
            "mannwhitney_p":    round(float(p_val), 6) if not np.isnan(p_val) else "n/a",
        }

        print(f"\n  {feat}:")
        print(f"    Median  CONFIRMED={med_conf:.4f}   FP={med_fp:.4f}")
        print(f"    Std     CONFIRMED={std_conf:.4f}   FP={std_fp:.4f}")
        print(f"    Mann-Whitney p = {p_str}")

    return obs


# ===========================================================================
# 16. SAVE PROCESSED DATA
# ===========================================================================
def save_processed_data(labeled_df: pd.DataFrame,
                         candidate_df: pd.DataFrame) -> None:
    """Save labeled, candidate, and feature list to processed/."""
    print("\n[INFO] Saving processed data...")

    # Labeled: features + target + metadata
    cols_to_save = FEATURE_COLUMNS + ["target", "koi_disposition", "kepid"]
    cols_to_save = [c for c in cols_to_save if c in labeled_df.columns]
    labeled_df[cols_to_save].to_csv(DATA_PROCESSED / "koi_labeled.csv", index=False)
    print(f"  Saved labeled   : {DATA_PROCESSED / 'koi_labeled.csv'} ({len(labeled_df):,} rows)")

    # Candidates: features only -- NO target column
    cand_cols = [c for c in FEATURE_COLUMNS + ["koi_disposition", "kepid"]
                 if c in candidate_df.columns]
    candidate_df[cand_cols].to_csv(DATA_PROCESSED / "koi_candidates.csv", index=False)
    print(f"  Saved candidates: {DATA_PROCESSED / 'koi_candidates.csv'} ({len(candidate_df):,} rows)")

    # Feature column list as JSON
    with open(DATA_PROCESSED / "feature_columns.json", "w") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)
    print(f"  Saved features  : {DATA_PROCESSED / 'feature_columns.json'}")


# ===========================================================================
# 17. GENERATE REPORT
# ===========================================================================
def generate_report(
    df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    balance_stats: dict,
    miss_df: pd.DataFrame,
    skew_df: pd.DataFrame,
    corr: pd.DataFrame,
    sep_obs: dict,
) -> None:
    """Write the Phase 1 EDA Markdown report using actual computed dataset values."""
    print("\n[INFO] Writing Phase 1 EDA report...")

    n_total   = len(df)
    n_cols    = df.shape[1]
    n_labeled = len(labeled_df)
    n_cand    = len(candidate_df)

    conf_n   = balance_stats["confirmed_n"]
    fp_n     = balance_stats["fp_n"]
    conf_pct = balance_stats["confirmed_pct"]
    fp_pct   = balance_stats["fp_pct"]
    imbal    = balance_stats["imbalanced"]

    # Correlation notable pairs (|r| > 0.5)
    pairs = []
    cols_c = corr.columns.tolist()
    for i in range(len(cols_c)):
        for j in range(i + 1, len(cols_c)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                pairs.append((cols_c[i], cols_c[j], round(r, 3)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Top 5 skewed features
    top_skew   = skew_df.head(5)[["feature", "skewness"]].to_dict("records")
    high_miss  = miss_df[miss_df["missing_pct"] > 20].to_dict("records")

    lines = [
        "# Phase 1 EDA Report",
        "## Kepler Exoplanet Classification \u2014 Explainable ML System",
        "",
        "*Generated from actual NASA Exoplanet Archive cumulative KOI table.*",
        "",
        "---",
        "",
        "## Dataset Overview",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Total rows | {n_total:,} |",
        f"| Total columns | {n_cols} |",
        f"| Labeled rows (CONFIRMED + FP) | {n_labeled:,} |",
        f"| Candidate rows | {n_cand:,} |",
        "| Source | NASA Exoplanet Archive cumulative KOI table |",
        "",
        "---",
        "",
        "## Target Distribution",
        "",
        "| Class | Count | Percentage |",
        "|-------|-------|------------|",
        f"| CONFIRMED (target=1) | {conf_n:,} | {conf_pct:.1f}% |",
        f"| FALSE POSITIVE (target=0) | {fp_n:,} | {fp_pct:.1f}% |",
        f"| CANDIDATE (no label) | {n_cand:,} | \u2014 |",
        "",
        "---",
        "",
        "## Leakage Columns Removed",
        "",
        "The following columns were removed prior to feature analysis because "
        "they are derived from or directly encode the disposition decision:",
        "",
    ]
    for c in LEAKAGE_COLUMNS:
        lines.append(f"- `{c}`")

    lines += [
        "",
        "---",
        "",
        "## Missingness",
        "",
        "| Feature | Missing Count | Missing % |",
        "|---------|--------------|-----------|",
    ]
    for _, row in miss_df.iterrows():
        lines.append(
            f"| {row['feature']} | {int(row['missing_count']):,} | {row['missing_pct']:.1f}% |"
        )

    if high_miss:
        lines += ["", "**Features exceeding 20% missing:**", ""]
        for h in high_miss:
            lines.append(f"- `{h['feature']}` \u2014 {h['missing_pct']:.1f}% missing")
        lines.append(
            "\nThese features are flagged for review in Phase 2 "
            "(imputation strategy decision required)."
        )
    else:
        lines += ["", "No features exceed 20% missing."]

    lines += [
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "See `reports/phase_1/summary_statistics.csv` for full descriptive statistics "
        "(count, mean, std, min, 25%, median, 75%, max, skewness) for all 11 features.",
        "",
        "---",
        "",
        "## Skewness",
        "",
        "| Feature | Skewness |",
        "|---------|----------|",
    ]
    for row in top_skew:
        lines.append(f"| {row['feature']} | {row['skewness']:.2f} |")

    lines += [
        "",
        "Features with |skewness| > 1 are candidates for log or power transformation "
        "in Phase 2. No transformations are applied in Phase 1.",
        "",
        "---",
        "",
        "## Correlation Analysis",
        "",
        "Pearson correlation was computed for all 11 features on the labeled dataset.",
        "",
        "**Notable pairs (|r| > 0.5):**",
        "",
    ]
    if pairs:
        lines += ["| Feature A | Feature B | r |", "|-----------|-----------|---|"]
        for a, b, r in pairs:
            lines.append(f"| {a} | {b} | {r:.3f} |")
    else:
        lines.append("No pairs exceed |r| = 0.5.")

    lines += [
        "",
        "No features are automatically removed based on correlation alone; "
        "decisions will be made in Phase 2 with domain context.",
        "",
        "---",
        "",
        "## Distribution Analysis",
        "",
        "Histogram and KDE plots were generated for: "
        + ", ".join(f"`{f}`" for f in DIST_FEATURES) + ".",
        "",
        "See `plots/phase_1/dist_*.png` for visualisations.",
        "",
        "---",
        "",
        "## Class Separability",
        "",
        "| Feature | Median CONFIRMED | Median FP | Mann-Whitney p |",
        "|---------|-----------------|-----------|----------------|",
    ]
    for feat, o in sep_obs.items():
        lines.append(
            f"| {feat} | {o['median_confirmed']} | {o['median_fp']} | {o['mannwhitney_p']} |"
        )

    lines += [
        "",
        "Mann-Whitney p-values indicate whether the class distributions differ "
        "statistically. A small p-value suggests different distributions, "
        "but does not guarantee predictive separability on its own.",
        "",
        "---",
        "",
        "## Data Quality Issues",
        "",
        "1. **Missingness**: Some features have missing values; imputation strategy "
        "   to be determined in Phase 2.",
        "2. **Skewness**: Several features are heavily right-skewed; transformation "
        "   candidates flagged for Phase 2.",
        "3. **Outliers**: Extreme values present in period, depth, and radius; "
        "   clipping/Winsorising strategy deferred to Phase 2.",
        "4. **Class imbalance**: "
        + ("Present \u2014 will require class-weighting or resampling in Phase 2."
           if imbal else "Mild or absent \u2014 standard treatment adequate."),
        "",
        "---",
        "",
        "## Physical Interpretation",
        "",
        "This section provides cautious physical context for the observed feature "
        "distributions based on transit photometry principles.",
        "",
        "- **Transit depth (`koi_depth`)**: Measures the fractional flux decrease "
        "  during transit. Planets produce characteristically shallower, "
        "  more consistent depths than eclipsing binaries. Differences in depth "
        "  distributions between classes may reflect this, but overlap is expected.",
        "",
        "- **Planetary radius (`koi_prad`)**: Derived from transit depth and stellar "
        "  radius. True exoplanets cluster at sub-Neptune to Jupiter radii; "
        "  very large inferred radii often indicate false positives "
        "  (diluted eclipsing binaries). Observed class differences are consistent "
        "  with this, but EDA alone does not confirm exoplanet status.",
        "",
        "- **Transit duration (`koi_duration`)**: Depends on orbital velocity and "
        "  stellar radius. Extremely short or long durations can indicate "
        "  non-planetary scenarios. Both classes show broad distributions.",
        "",
        "- **Orbital period (`koi_period`)**: Kepler's sampling window favours "
        "  shorter-period planets. The distribution reflects both detection bias "
        "  and physical occurrence rates; false positives can occur at any period.",
        "",
        "- **Signal-to-noise ratio (`koi_model_snr`)**: Higher SNR signals tend to "
        "  be genuine transits. Confirmed planets are expected to show higher "
        "  median SNR than false positives.",
        "",
        "- **Stellar radius (`koi_srad`)**: Affects derived planetary radius and "
        "  transit depth interpretation. Uncertainty in stellar parameters "
        "  propagates to planetary parameters.",
        "",
        "**Important caveat**: These physical interpretations are consistent with "
        "known transit science but cannot be confirmed from EDA alone. "
        "The ML model in later phases will learn from the joint feature space "
        "rather than any single physical argument.",
        "",
        "---",
        "",
        "## Phase 1 Conclusion",
        "",
        f"- Dataset loaded: **{n_total:,} KOI objects** across **{n_cols} columns**",
        f"- Labeled for modelling: **{n_labeled:,}** ({conf_n:,} CONFIRMED, {fp_n:,} FALSE POSITIVE)",
        f"- Held out for prediction: **{n_cand:,} CANDIDATEs**",
        "- All 11 feature columns validated as present",
        "- All 6 leakage columns removed",
        "- Missingness, skewness, and correlation quantified (no transformations applied)",
        "- Processed datasets and feature list saved",
        "",
        "**Phase 1 is complete. Phase 2 will handle: imputation, "
        "transformation, train/test split, and baseline modelling.**",
    ]

    report_path = REPORTS_P1 / "phase_1_eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {report_path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    print("=" * 70)
    print("PHASE 1: DATA ACQUISITION & EXPLORATORY DATA ANALYSIS")
    print("Kepler Exoplanet Classification")
    print("=" * 70)

    # 1. Download dataset if needed
    download_dataset()

    # 2. Load
    df = load_data()

    # 3. Audit
    audit_dataset(df)

    # 4. Create target datasets
    labeled_df, candidate_df = create_target_datasets(df)

    # 5. Remove leakage columns
    labeled_df, candidate_df = remove_leakage_columns(labeled_df, candidate_df)

    # 6. Validate all 11 feature columns present
    validate_features(labeled_df)

    # 7. Class balance
    balance_stats = analyze_class_balance(labeled_df)

    # 8. Missingness
    miss_df = analyze_missingness(labeled_df)

    # 9. Summary statistics
    generate_summary_statistics(labeled_df)

    # 10. Skewness
    skew_df = analyze_skewness(labeled_df)

    # 11. Distribution plots (histogram + KDE)
    generate_distribution_plots(labeled_df)

    # 12. Boxplots
    generate_boxplots(labeled_df)

    # 13. Correlation heatmap
    corr = generate_correlation_heatmap(labeled_df)

    # 14. Quantile analysis
    analyze_quantiles(labeled_df)

    # 15. Class separability (Mann-Whitney U)
    sep_obs = analyze_class_separability(labeled_df)

    # 16. Save processed datasets
    save_processed_data(labeled_df, candidate_df)

    # 17. Write EDA report
    generate_report(
        df, labeled_df, candidate_df,
        balance_stats, miss_df, skew_df, corr, sep_obs,
    )

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
