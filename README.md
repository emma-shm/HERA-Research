# HERA-Research
Repository for CW calibrations and data analysis scripts for the HERA collaboration and the HASP flight


### `Calibration/`
Scripts for calibrating HERA's CW detectors + using calibration for particle identification
- Parsing raw CosmicWatch serial/datalogger output.
- Fitting the ADC energy-deposition spectrum (Moyal/Landau) and extracting the MPV for gain calibration.
- Deadtime correction and livetime normalization.
- Temperature-dependent calibration.

### `HASP-Drexel/`
Scripts for applying that code/analysis to HASP flight data
- Ingesting and cleaning the flight datalogger files.
- Applying the calibration from `Calibration/`.
- Coincidence-rate and flux computation vs. altitude and time.
- 2D density heatmaps and summary plots (Plotly / Matplotlib).
- *(add your specific notebook names here)*

---

## Getting started

### Prerequisites
- Python 3.10+
- Jupyter (Notebook or Lab)

### Clone the repository
```bash
git clone https://github.com/emma-shm/HERA-Research.git
cd HERA-Research
```

### Set up an environment
Using `venv`:
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or using conda:
```bash
conda create -n hera python=3.11
conda activate hera
pip install -r requirements.txt
```

### Suggested `requirements.txt`
If you don't already have one, this covers the typical stack for this analysis:
```
numpy
scipy
pandas
matplotlib
plotly
lmfit
jupyter
tqdm
```

### Launch
```bash
jupyter lab        # or: jupyter notebook
```
Open the notebook you want under `Calibration/` or `HASP-Drexel/` and run the cells top to bottom.

---

## Data

The notebooks expect raw output from the CosmicWatch datalogger (SD-card / serial logs). Each event row typically includes an event number, timestamp/millis, the measured ADC value (SiPM peak), a computed deadtime, and — depending on configuration — coincidence and temperature fields.

Large raw data files and generated outputs are kept out of version control via `.gitignore`. If you want to reproduce the analysis, place the flight/calibration data in the location the notebook points to (update the path at the top of the notebook as needed).

---

## Analysis notes

A few of the core techniques used across the notebooks:

- **MPV extraction via Moyal/Landau fit** — the energy deposited by a minimum-ionizing particle in a thin scintillator follows a Landau-like distribution; fitting it gives a stable most-probable value for gain calibration rather than relying on the mean.
- **Deadtime & livetime correction** — the measured rate is corrected for the time the detector spends processing each event so that flux estimates reflect true livetime.
- **Coincidence rate** — requiring simultaneous hits in stacked detectors suppresses noise and singles, isolating the through-going cosmic-ray/muon component.
- **Temperature-dependent calibration** — SiPM gain drifts with temperature, which matters over a balloon ascent; the calibration accounts for this so in-flight ADC values map correctly to energy.
- **Altitude profile** — plotting corrected coincidence rate vs. altitude recovers the Regener–Pfotzer curve.

---

## Citation

If you use this code or reference this work, please cite the associated thesis:

> Martignoni, E. *High-Altitude Engineering and Analysis for Cosmic Ray Tomography.* M.S. Thesis, Department of Mechanical Engineering, Drexel University, 2026.

---

## License

*No license file is currently included.* Without a license, default copyright applies and others may not reuse the code. If you'd like to allow reuse, add a `LICENSE` file — [MIT](https://choosealicense.com/licenses/mit/) is a common, permissive choice for research code.

---

## Contact

Maintained by [@emma-shm](https://github.com/emma-shm). Questions, issues, or contributions are welcome via the repo's [Issues](https://github.com/emma-shm/HERA-Research/issues) page.


