"""
TESS Exoplanet Transit Light Curve Analysis
============================================
Analyzes photometric time-series data from the TESS mission using Lightkurve.
Extracts transit signals via detrending and fits planetary parameters using batman.

Target: WASP-39b (a well-known hot Jupiter; great for validation)
TIC ID: 350318537
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import batman
from scipy.optimize import minimize
from scipy.signal import savgol_filter


# ─────────────────────────────────────────────
# 1. DOWNLOAD & INSPECT TESS LIGHT CURVE
# ─────────────────────────────────────────────

def download_lightcurve(target="WASP-39", mission="TESS", sector=None):
    """
    Download a TESS light curve using Lightkurve.
    Returns a stitched LightCurve object.
    """
    print(f"[1/5] Searching TESS data for {target}...")
    search = lk.search_lightcurve(target, mission=mission, author="SPOC", exptime=120)
    print(f"      Found {len(search)} sector(s): {search.table['sector'].tolist()}")

    if len(search) == 0:
        raise ValueError(f"No TESS data found for {target}")

    # Download first available sector (or specified)
    idx = 0 if sector is None else search.table["sector"].tolist().index(sector)
    lc = search[idx].download()
    print(f"      Downloaded Sector {search.table['sector'][idx]}  |  "
          f"{len(lc)} cadences  |  Baseline: {lc.time.value[-1] - lc.time.value[0]:.1f} days")
    return lc


# ─────────────────────────────────────────────
# 2. PREPROCESSING & DETRENDING
# ─────────────────────────────────────────────

def preprocess(lc):
    """
    Clean and detrend the light curve:
      - Remove NaNs and flagged cadences
      - Normalize flux to median
      - Sigma-clip outliers (5σ)
      - Remove long-term systematics with Savitzky-Golay filter
    """
    print("[2/5] Preprocessing light curve...")

    # Use SAP or PDCSAP flux; prefer PDCSAP (already partly corrected)
    lc = lc.select_flux("pdcsap_flux")

    # Remove NaN and quality-flagged points
    lc = lc.remove_nans().remove_outliers(sigma=5)

    # Normalize
    lc = lc.normalize()

    # Flatten with Savitzky-Golay to remove residual instrumental trends
    # window_length must be odd and shorter than any expected transit duration
    lc_flat, trend = lc.flatten(window_length=401, return_trend=True)

    print(f"      Remaining cadences after cleaning: {len(lc_flat)}")
    print(f"      Median normalized flux: {np.median(lc_flat.flux.value):.6f}")
    print(f"      RMS scatter: {np.std(lc_flat.flux.value)*1e6:.1f} ppm")

    return lc_flat, lc, trend


# ─────────────────────────────────────────────
# 3. PERIOD SEARCH (BLS)
# ─────────────────────────────────────────────

def find_period(lc_flat):
    """
    Box Least Squares (BLS) periodogram to identify transit period.
    """
    print("[3/5] Running BLS periodogram to find transit period...")

    pg = lc_flat.to_periodogram(method="bls",
                                period=np.linspace(0.5, 15, 10000),
                                duration=[0.05, 0.1, 0.15, 0.2])
    best_period = pg.period_at_max_power.value
    best_t0     = pg.transit_time_at_max_power.value
    best_dur    = pg.duration_at_max_power.value
    depth       = pg.depth_at_max_power

    print(f"      Best period:   {best_period:.4f} days")
    print(f"      Transit epoch: {best_t0:.4f} BTJD")
    print(f"      Duration:      {best_dur*24:.2f} hours")
    print(f"      Depth:         {depth*1e6:.0f} ppm  →  Rp/Rs ≈ {np.sqrt(depth):.4f}")

    return pg, best_period, best_t0, best_dur


# ─────────────────────────────────────────────
# 4. BATMAN TRANSIT MODEL FIT
# ─────────────────────────────────────────────

def batman_model(t, t0, period, rp, a, inc, u1, u2):
    """
    Compute batman transit light curve for given parameters.

    Parameters
    ----------
    t      : array  – time array (days)
    t0     : float  – mid-transit time (days)
    period : float  – orbital period (days)
    rp     : float  – planet-to-star radius ratio (Rp/Rs)
    a      : float  – scaled semi-major axis (a/Rs)
    inc    : float  – orbital inclination (degrees)
    u1, u2 : float  – quadratic limb-darkening coefficients
    """
    params = batman.TransitParams()
    params.t0    = t0
    params.per   = period
    params.rp    = rp
    params.a     = a
    params.inc   = inc
    params.ecc   = 0.0       # circular orbit assumption
    params.w     = 90.0      # longitude of periastron (irrelevant for circular)
    params.limb_dark = "quadratic"
    params.u     = [u1, u2]

    m = batman.TransitModel(params, t)
    return m.light_curve(params)


def fit_transit(lc_flat, period, t0, duration):
    """
    Fit transit model using scipy minimize (Nelder-Mead).
    Returns best-fit parameters and fold data.
    """
    print("[4/5] Fitting batman transit model...")

    # Fold and bin the light curve
    lc_fold = lc_flat.fold(period=period, epoch_time=t0)

    # Only fit points within ±2× transit duration of mid-transit
    mask = np.abs(lc_fold.time.value) < 2 * duration
    t_fit = lc_fold.time.value[mask]
    f_fit = lc_fold.flux.value[mask]
    e_fit = lc_fold.flux_err.value[mask] if lc_fold.flux_err is not None else np.ones_like(f_fit) * 1e-4

    # Initial guesses (WASP-39b literature values as starting point)
    p0 = dict(t0=0.0, period=period, rp=0.145, a=11.4, inc=87.8, u1=0.4, u2=0.2)

    def neg_log_likelihood(x):
        rp, a, inc, u1, u2 = x
        if rp < 0 or rp > 1 or a < 1 or inc < 50 or inc > 90:
            return 1e10
        model = batman_model(t_fit, 0.0, period, rp, a, inc, u1, u2)
        residuals = (f_fit - model) / e_fit
        return 0.5 * np.sum(residuals**2)

    x0 = [p0["rp"], p0["a"], p0["inc"], p0["u1"], p0["u2"]]
    result = minimize(neg_log_likelihood, x0, method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-8})

    rp_fit, a_fit, inc_fit, u1_fit, u2_fit = result.x

    print(f"      Rp/Rs      = {rp_fit:.4f}  →  Rp ≈ {rp_fit * 1.279:.3f} RJup  (assuming WASP-39 R★)")
    print(f"      a/Rs       = {a_fit:.2f}")
    print(f"      Inclination= {inc_fit:.2f}°")
    print(f"      u1, u2     = {u1_fit:.3f}, {u2_fit:.3f}")
    print(f"      Transit depth = {rp_fit**2 * 1e6:.0f} ppm")

    best_fit = dict(t0=0.0, period=period, rp=rp_fit, a=a_fit,
                    inc=inc_fit, u1=u1_fit, u2=u2_fit)
    return best_fit, lc_fold, t_fit, f_fit


# ─────────────────────────────────────────────
# 5. PLOTTING
# ─────────────────────────────────────────────

def make_plots(lc_raw, lc_flat, trend, pg, lc_fold, best_fit, t_fit, f_fit):
    """
    Generate a 4-panel summary figure.
    """
    print("[5/5] Generating plots...")

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    panel_kw = dict(facecolor="#161b22")
    spine_color = "#30363d"
    txt_color   = "#e6edf3"
    accent      = "#58a6ff"
    data_color  = "#79c0ff"
    trend_color = "#f78166"

    # ── Panel A: Raw + trend ──────────────────
    ax1 = fig.add_subplot(gs[0, 0], **panel_kw)
    ax1.scatter(lc_raw.time.value, lc_raw.flux.value,
                s=0.5, c=data_color, alpha=0.4, rasterized=True)
    ax1.plot(trend.time.value, trend.flux.value, color=trend_color,
             lw=1.5, label="SG trend", zorder=5)
    ax1.set_xlabel("Time (BTJD)", color=txt_color)
    ax1.set_ylabel("Normalized Flux", color=txt_color)
    ax1.set_title("A  Raw Light Curve + Systematic Trend", color=txt_color, loc="left", fontsize=10)
    ax1.legend(facecolor=spine_color, labelcolor=txt_color, fontsize=8)
    _style_ax(ax1, txt_color, spine_color)

    # ── Panel B: Detrended ────────────────────
    ax2 = fig.add_subplot(gs[0, 1], **panel_kw)
    ax2.scatter(lc_flat.time.value, lc_flat.flux.value,
                s=0.5, c=data_color, alpha=0.4, rasterized=True)
    ax2.axhline(1.0, color=accent, lw=0.8, ls="--", alpha=0.6)
    ax2.set_xlabel("Time (BTJD)", color=txt_color)
    ax2.set_ylabel("Normalized Flux", color=txt_color)
    ax2.set_title("B  Detrended & Cleaned Light Curve", color=txt_color, loc="left", fontsize=10)
    _style_ax(ax2, txt_color, spine_color)

    # ── Panel C: BLS periodogram ──────────────
    ax3 = fig.add_subplot(gs[1, 0], **panel_kw)
    ax3.plot(pg.period.value, pg.power.value, color=accent, lw=0.8)
    best_p = pg.period_at_max_power.value
    ax3.axvline(best_p, color=trend_color, lw=1.5, ls="--",
                label=f"P = {best_p:.4f} d")
    ax3.set_xlabel("Period (days)", color=txt_color)
    ax3.set_ylabel("BLS Power", color=txt_color)
    ax3.set_title("C  BLS Periodogram", color=txt_color, loc="left", fontsize=10)
    ax3.legend(facecolor=spine_color, labelcolor=txt_color, fontsize=8)
    _style_ax(ax3, txt_color, spine_color)

    # ── Panel D: Phase-folded + batman fit ────
    ax4 = fig.add_subplot(gs[1, 1], **panel_kw)

    # All folded data (grey)
    ax4.scatter(lc_fold.time.value, lc_fold.flux.value,
                s=0.6, c="#484f58", alpha=0.3, rasterized=True)

    # Binned
    bins   = np.linspace(lc_fold.time.value.min(), lc_fold.time.value.max(), 200)
    binned = lc_fold.bin(time_bin_size=0.01)
    ax4.scatter(binned.time.value, binned.flux.value,
                s=8, c=data_color, zorder=4, label="Binned data")

    # Batman model
    t_model = np.linspace(t_fit.min(), t_fit.max(), 1000)
    f_model = batman_model(t_model, **best_fit)
    ax4.plot(t_model, f_model, color=trend_color, lw=2, zorder=5,
             label=f"batman fit  Rp/Rs={best_fit['rp']:.4f}")

    ax4.set_xlabel("Phase (days from mid-transit)", color=txt_color)
    ax4.set_ylabel("Normalized Flux", color=txt_color)
    ax4.set_title("D  Phase-Folded Transit + batman Model", color=txt_color, loc="left", fontsize=10)
    ax4.legend(facecolor=spine_color, labelcolor=txt_color, fontsize=8)
    _style_ax(ax4, txt_color, spine_color)

    fig.suptitle("TESS Exoplanet Transit Analysis  ·  WASP-39b",
                 color=txt_color, fontsize=14, y=0.98)

    plt.savefig("transit_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("      Saved → transit_analysis.png")
    plt.show()


def _style_ax(ax, txt_color, spine_color):
    ax.tick_params(colors=txt_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
    ax.xaxis.label.set_color(txt_color)
    ax.yaxis.label.set_color(txt_color)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TARGET = "WASP-39"   # Well-characterised hot Jupiter; JWST Early Release target

    lc_raw              = download_lightcurve(TARGET)
    lc_flat, lc_raw_n, trend = preprocess(lc_raw)
    pg, period, t0, dur = find_period(lc_flat)
    best_fit, lc_fold, t_fit, f_fit = fit_transit(lc_flat, period, t0, dur)
    make_plots(lc_raw_n, lc_flat, trend, pg, lc_fold, best_fit, t_fit, f_fit)

    print("\n✓ Analysis complete.")
    print(f"  Period  = {best_fit['period']:.5f} d  (literature: 4.05527 d)")
    print(f"  Rp/Rs   = {best_fit['rp']:.4f}       (literature: 0.1454)")
