# 🔭 TESS Exoplanet Transit Light Curve Analysis

Automated pipeline for detecting and characterizing exoplanet transits using photometric time-series data from NASA's **Transiting Exoplanet Survey Satellite (TESS)**.

**Target: WASP-39b** — a hot Jupiter at ~215 pc, one of the first exoplanets characterized by JWST's Early Release Science program.

---

## Pipeline

```
TESS MAST Archive
      ↓  Lightkurve (SPOC PDCSAP flux)
Raw Light Curve
      ↓  Sigma-clipping + Savitzky–Golay detrending
Cleaned Light Curve
      ↓  Box Least Squares (BLS) periodogram
Orbital Period + Epoch
      ↓  Phase-folding + batman transit model
Fitted Parameters (Rp/Rs, a/Rs, inclination, limb darkening)
```

---

## Results

| Parameter | This Analysis | Literature |
|-----------|:---:|:---:|
| Orbital Period (days) | 4.0553 | 4.05528 |
| Rp/Rs | 0.1451 | 0.1454 |
| a/Rs | 11.38 | 11.55 |
| Inclination (°) | 87.6 | 87.83 |
| Transit Depth (ppm) | ~2108 | ~2113 |

---

## Techniques

- **Lightkurve** SPOC pipeline access via MAST archive (NASA)
- **PDCSAP flux** selection (Pre-search Data Conditioning Simple Aperture Photometry)
- **Savitzky–Golay filtering** (window = 401 cadences / ~13 hrs) for systematic detrending
- **5σ outlier rejection** via iterative sigma-clipping
- **Box Least Squares (BLS)** periodogram for blind period search (0.5–15 days)
- **batman** transit model fitting with Nelder–Mead optimization
  - Free parameters: Rp/Rs, a/Rs, inclination, u1, u2 (quadratic limb darkening)
  - Fixed: circular orbit (e=0), mid-transit time anchored to BLS epoch

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/tess-exoplanet-analysis
cd tess-exoplanet-analysis
pip install -r requirements.txt
```

**Run the script:**
```bash
python tess_transit_analysis.py
```

**Or open the notebook:**
```bash
jupyter notebook tess_transit_notebook.ipynb
```

> Requires internet connection to download TESS data from NASA MAST on first run (~50 MB).

---

## Output

Running the pipeline produces `transit_analysis.png` — a 4-panel summary figure:

| Panel | Content |
|-------|---------|
| A | Raw TESS light curve with fitted systematic trend |
| B | Detrended & sigma-clipped light curve |
| C | BLS periodogram with detected period |
| D | Phase-folded transit + batman model overlay |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `lightkurve` | TESS/Kepler data access & preprocessing |
| `batman-package` | Analytic transit light curve models |
| `numpy` / `scipy` | Numerical optimization |
| `matplotlib` | Visualization |
| `astropy` | Time/coordinate handling |

---

## References

- Kreidberg (2015) — *batman: BAsic Transit Model cAlculatioN in Python*. PASP, 127, 1161
- Lightkurve Collaboration (2018) — *Lightkurve: Kepler and TESS time series analysis in Python*
- Faedi et al. (2011) — WASP-39b discovery paper. A&A, 531, A40
- Rustamkulov et al. (2023) — WASP-39b JWST Early Release Science. Nature, 614, 659
