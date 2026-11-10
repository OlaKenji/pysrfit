# pySRFit

A modern, interactive GUI application for analysing muon spin rotation, relaxation and resonance (μSR) data. pySRFit is inspired by musrfit and uses a similar parameter and THEORY style.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### Core Capabilities
- **Multi-beamline support**: Load raw μSR data from PSI, ISIS, J-PARC and TRIUMF.
- **Fitting**: Perform single-run, global and sequential fits using MINUIT.
- **Fourier transform**: Display data and fits in the time domain or as Fourier transforms.
- **Save dataset**: Save complete analyses as `.pysrfit` projects.
- **Generate a tabulated result table**: Collect fitted projects into CSV result tables.
- **Custom relaxation functions**: Add custom Python relaxation functions.

---

## Installation

```bash
pip install pysrfit

# Launch the GUI
pysrfit
```

## Usage

### Basic Workflow

1. **Launch the application**:
   ```bash
   pysrfit
   ```

2. **Load data**:
   - Configure the selected run in **Runs / Data**.

3. **Model**:
   - Define the parameters and THEORY model.

4. **Fit and visualise**:
   - Apply the parameters, fit the data, and save the `.pysrfit` project.

### Supported Facilities

- **PSI:** MusrRoot `.root` files, tested with GPS, GPD, FLAME, HAL and LEM.
- **ISIS:** NeXus `.nxs` files.
- **J-PARC:** NeXus `.nxs` files, tested with MS1 and MD1.
- **TRIUMF:** binary MUD `.msr` and `.mud` files, tested with M20 and M15.

## Theory

Consecutive THEORY lines multiply, while `+` starts an additive component:

```text
asymmetry 1
simplExpo 2
TFieldCos 3 4
+
asymmetry 5
```

Built-in functions include exponential, stretched-exponential and Gaussian relaxation, cosine and Bessel oscillations, Gaussian and Lorentzian Kubo–Toyabe models, and their longitudinal-field and dynamic forms. Additional models can be registered with Python in **Custom models**.

### Generalized two-state collision model

The two-state model from
[O. K. Forslund, Phys. Rev. B **112**, 024309 (2025)](https://doi.org/10.1103/wf5z-lj1c)
accepts any registered static function:

```text
twoState nu1 nu2 g | staticFunction1 ... | staticFunction2 ...
```

For example, symmetric switching between two Gaussian Kubo–Toyabe states is:

```text
asymmetry 1
twoState 2 2 3 | statGssKt 4 | statGssKt 5
```

The first two references are the switching rates. The third is the probability of starting in state 1 and should be constrained between 0 and 1.

---
## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---
## License

This project is licensed under the MIT License.

---
## Citation

If you use pySRFit in your research, please cite:

```bibtex

```

---
