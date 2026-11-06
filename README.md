# mysr — GUI-first prototype (v0.2.0)

An extensible desktop application for μSR analysis with modular internals. Its GUI is built using PyQt5 and an embedded Matplotlib canvas. Fitting uses `iminuit` (the MINUIT2 C++ engine). This is a **functional starter project**, **not** a validated musrfit replacement.

## Installation

Python 3.10+ is required. Install the project in a clean virtual environment:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
# Optional raw-data readers (MusrRoot and NeXus/HDF5):
python -m pip install '.[data]'
```

For development, use an editable install with the test dependencies:

```bash
python -m pip install -e '.[dev]'
```

Launch the installed GUI from the `mysr` application shortcut/launcher. From a source checkout, it can also be started from a terminal:

```bash
python run.py
```

`run.py` is only a thin GUI launcher. Open projects and data files from the **File** menu; the application does not provide command-line analysis options.

From the GUI, **File → Open project / .msr in new tab…** accepts `.msr` (import) and `.mysr` (native JSON project). **Save current project…** writes `.mysr` and never overwrites the original `.msr`. The original `.msr` text and unknown sections are retained in the saved native project.

Data files remain external. A `.mysr` project stores the source path and the reusable detector/t0/background/packing configuration, not raw histogram arrays. The active source is loaded when selected, plotted, or fitted, cached in memory, and automatically reloaded if its path, size, or modification time changes. **Change data source…** keeps the analysis configuration; **Open data in new analysis…** creates a separate tab.

To exercise fitting, choose **Open data in new analysis…** and open `examples/demo_spectrum.csv`, which contains time (μs), asymmetry, and its one-sigma uncertainty. Fitting needs the `iminuit` dependency.

## Existing prototype features

- Multiple independent analysis tabs, each with its own project, data, plot and fit state. Each analysis contains pages for Parameters, Theory, Runs/Data, Custom models, and the original `.msr`.
- A THEORY function picker plus Runs/Data controls for fit type, detector groups, per-detector t0, background/data channels, and packing.
- A raw-histogram correction dialog for selecting each detector and placing t0 (`t`), data start/end (`d`/`D`), and background start/end (`b`/`B`) directly with the pointer.
- Free-text parameter editing with preserved comments/blank lines, sequential renumbering, left-aligned formatting, selectable significant digits, fixed parameters (`step=0`), parameter bounds, and line-numbered validation.
- New parameter editors default to four significant digits, and newly configured raw-data runs default to packing 20; imported project settings remain unchanged.
- Model preview, Matplotlib navigation/zoom, residuals, PNG export, and an exploratory FFT view.
- Import of the accompanying `examples/ZF44K.msr` including all 23 parameters, shared phase references, seven summed components, PSI-BIN run *metadata*, commands, plot/FFT settings and saved statistics. Only part of these imported settings is *executed*. Remaining source data is retained for future support.
- musrfit-style THEORY subset: `asymmetry`, `simplExpo`, `simpleGss`, `TFieldCos`, `statGssKt`, and aliases `a`, `se`, `sg`, `tf`. Consecutive THEORY lines multiply; an isolated `+` line adds components.
- Simple FUNCTIONS syntax including `fun1 = gamma_mu * par5`, `mapN`, constants, and arithmetic; evaluated through restricted AST rather than Python `eval`.
- Model function registry, GUI-editable Python model code activated *explicitly* by the user. No recompilation.
- MINUIT2 chi-square fitting to fit-ready asymmetry or musrfit type-0 histogram counts. Type 0 uses `N0 * exp(-t/tau_mu) * (1 + A(t)) + background`; invalid minima do not replace model parameters.
- A modular reader registry with independent readers for 3-column ASCII, a strict HDF5 NeXus `NXdata` subset, and PSI MusrRoot. MusrRoot imports `hDecayNNN` counts plus detector names, t0, first-good/last-good bins, and run/sample metadata.
- musrfit-style detector grouping (`1 2 3` or `1-3`), integer-bin t0 alignment, background correction, packing, Poisson propagation, and both type-2 asymmetry and type-0 raw-count preparation.
- Native `.mysr` JSON read/write; data file paths remain external rather than embedding massive raw arrays.

## Important scientific / compatibility limitations

1. **The attached `.msr` file does not contain raw histogram counts.** It refers to `./Data/deltat_tdc_gps_2651` (PSI-BIN). Without the corresponding `.bin` file and a validated PSI-BIN reader, the application correctly displays a **model preview only**. It must not claim to have fitted the measurement.
2. **PSI-BIN, PSI-MDU, TRIUMF-MUD, and facility-specific ISIS/J-PARC NeXus layouts are not implemented yet.** PSI MusrRoot is implemented and tested against the included GPS fixture. Unknown formats fail visibly.
3. The initial MSR parser is **one-way for fitting metadata**. Native projects preserve original text but saving changes to `.mysr` does **not** export an edited `.msr` file. The imported `GLOBAL`, `FOURIER`, `COMMANDS`, `PLOT`, and `STATISTIC` blocks are retained but not comprehensively executed.
4. `FUNCTIONS` implements basic arithmetic, not all musrfit expressions, and THEORY currently implements only five primitive types. Multi-run global fitting is not yet implemented, although multiple RUN blocks are retained and parsed. At present, the first RUN block is used for mapping/data processing.
5. Histogram grouping currently requires integer-bin t0; fractional-bin interpolation, accelerator-cycle background rules, add-runs, and maximum-likelihood objectives remain to be implemented and validated against musrfit. Floating alpha is deliberately refused for precomputed asymmetry because it requires rebuilding the data and errors during every objective evaluation.
6. The FFT panel is exploratory (Hann window, FFT magnitude) and does **not** yet reproduce musrfit's FOURIER settings. There is no dynamic Kubo–Toyabe or two-state strong-collision model in this package yet.
7. **The custom model code is arbitrary Python.** Do not activate code from an untrusted project. Loading an `.msr` or `.mysr` never executes embedded custom code automatically.
8. The Python prototype has not been performance benchmarked or GUI-tested on each operating system. The fitting backend is MINUIT2 C++ via `iminuit` but model evaluation time may dominate complex fits.

## Adding your own theory function

Open the **Custom models** tab. It contains a ready-to-edit example:

```python
import numpy as np
from mysr.core.functions import register_function

@register_function("stretchedExpo", 2, "exp(-(lambda*t)**beta)", custom=True)
def stretched_expo(t, lam, beta):
    return np.exp(-(lam * np.asarray(t))**beta)
```

Click **Activate code** and explicitly confirm that you trust it. Then set the THEORY field to, for example:

```text
asymmetry     1
stretchedExpo 2 3
```

Function signature: `function(t_us: ndarray, *params: float) -> ndarray`. The returned array must match the shape of `t_us`.

The model from O. K. Forslund, *Physical Review B* **112**, 024309 (2025), “Two-state generalization of the strong collision model,” is an especially useful future custom-function example. **Do not treat a generic two-state approximation as the published model**: its exact equations and numerical limit cases need implementation and validation against the paper. The registry is set up so that a verified implementation can be dropped in as a standalone function.

## Adding a facility-specific raw-data reader

Implement a class with `can_load(path)` and `load(path)`, and register it:

```python
from mysr.io.loaders import register_loader
from mysr.io.base import HistogramRun

@register_loader
class MyFacilityReader:
    name = "My facility format"

    @staticmethod
    def can_load(path):
        return path.suffix.lower() == ".myraw"

    @staticmethod
    def load(path):
        # Decode the facility's real file layout and metadata here.
        # Must return time_us as a 1D array and counts as (detectors, bins).
        raise NotImplementedError("Implement from facility format specification")
```

For this initial prototype, import the module registering the loader into `mysr.io.loaders` so its registration runs. A real plugin entry-point discovery mechanism can come later. Detector mapping, t0 and correction then live in the independent `io/asymmetry.py` layer.

## Package layout

```text
mysr/
    app.py                 GUI application lifecycle
    session.py             project state and lazy runtime-data cache
    models.py              native project and parameter/run dataclasses
    msr.py                 non-destructive musrfit .msr importer
    core/
        functions.py       built-in + user function registry
        theory.py          THEORY and FUNCTIONS evaluation
        fit.py             MINUIT2 fitting bridge
    io/
        base.py            normalized spectrum/histogram interfaces
        loaders.py         auto-detection and data loader registry
        asymmetry.py       grouping, t0 alignment, type-0/type-2 preparation
        readers/
            musrroot.py    PSI MusrRoot reader
            nexus.py       strict generic NXdata reader
            text.py        reduced ASCII asymmetry reader
    gui/
        main_window.py     tabbed analysis workspaces and Matplotlib canvases
examples/
    ZF44K.msr              user-supplied example
    ZF44K.mysr             converted native example
    demo_spectrum.csv      synthetic fit-ready 3-column data
    make_demo_data.py      reproducible synthetic data generator
tests/
    test_session.py
    test_launcher.py
    test_gui.py
    test_msr.py
    test_theory.py
    test_io.py
    test_fit.py
```

## Testing

```bash
python -m pip install -e '.[dev]'
pytest -q
```

Tests cover the supplied MSR and PSI/GPS MusrRoot files, native JSON roundtripping, theory algebra, custom models, ASCII and NeXus reading, grouped detector/t0 alignment, type-0 histogram preparation, and asymmetry error propagation; MINUIT fit testing runs when `iminuit` is available.

## Licensing note

This prototype is original starter code supplied under MIT. However, **PyQt5 is licensed under GPL/commercial terms**, so distributing a complete app may impose additional obligations. For future permissive distribution, migrating the Qt binding to **PySide6 (LGPL)** deserves consideration. Reusing code from musrfit itself can separately introduce GPL-2.0-or-later obligations. Seek appropriate license review before packaging third-party code.

## Sensible next implementation milestones

1. Complete a validated PSI-BIN reader using a real `.bin` sample, and compare time axes and histogram counts with musrfit.
2. Validate t0, accelerator-cycle background, asymmetry, histogram fitting, and alpha conventions numerically against musrfit.
3. Add representative TRIUMF MUD and ISIS/J-PARC NeXus readers and fixtures through the existing reader registry.
4. Implement editable fit-function components / parameter coupling through GUI controls, and full multi-run fitting.
5. Add exact numerical/custom PRB 112, 024309 two-state model and benchmark it against established limiting cases.
6. Add CI wheels and GUI smoke tests on Windows, macOS and Linux, and benchmark NumPy vs Numba for expensive models.
