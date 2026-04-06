<div align="center">

# PlanetFinder

## [Online Demo](https://planetfinder.online)

### A neural network for exoplanet detection from stellar light curves
### [🇷🇺 Russian version here](https://github.com/Ztry8/PlanetFinder/blob/main/README_RU.md)

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76b900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![NASA Data](https://img.shields.io/badge/Data-NASA%20Kepler-0b3d91?style=for-the-badge&logo=nasa)](https://exoplanetarchive.ipac.caltech.edu/)

</div>

---

## Description

**PlanetFinder** is a fast, cross-platform machine learning system written in **Rust** that detects exoplanets using the transit method: by analyzing fluctuations in a star's brightness over time (light curves), the neural network predicts the number of planets in a stellar system.

The project uses real astronomical data from the **NASA Kepler** and **TESS** missions, accessed through the Python library `lightkurve`.

### Why does this exist?

**From an astronomical perspective** — it enables automatic analysis of millions of light curves that cannot be processed manually, accelerates the search for exoplanet candidates, and helps test hypotheses about transit paths and orbital periods.

**From a development perspective** — it is a practical example of applying LSTM to time-series analysis in Rust, a demonstration of working with real NASA scientific data, and a cross-platform project with optional GPU acceleration via CUDA.

---

## How it works

The method used by PlanetFinder is called **transit photometry**. When a planet passes in front of its star, it blocks some of the star's light — the star's brightness drops slightly. The number of planets can be determined from the depth, shape, and periodicity of these dips.

```
  Star brightness
  │
  │████████████████████████████████████████████████
  │                        ↓ planet transit
  │████████████████████████    ████████████████████
  │                       ░░░░░
  └──────────────────────────────────────────────▶ Time
                            └─ brightness drop ~1%
```

**Data processing pipeline:**

```
  NASA data (Kepler/TESS)
          │
          ▼
    download_data.py
    ├── Downloading light curves via lightkurve
    ├── Brightness normalization
    ├── Removing outliers and NaNs
    └── Saving to learn*.txt
          │
          ▼
    PlanetFinder (Rust)
    ├── Parsing .txt files
    ├── Time normalization [0..1]
    ├── Building tensors
    └── LSTM → Linear → Prediction
          │
          ▼
    Result: N planets in the system
```

---

## Neural network architecture

PlanetFinder uses **LSTM (Long Short-Term Memory)** — a recurrent neural network specifically designed to work with sequences and time series.

```
Input sequence (light curve)
[brightness₁, time₁] → [brightness₂, time₂] → ... → [brightnessₙ, timeₙ]
         │
         ▼
┌─────────────────────────────────────────────────┐
│             LSTM Layer (hidden states)           │
│   ┌──────┐     ┌──────┐           ┌──────┐      │
│   │ LSTM │ ──▶ │ LSTM │ ── ... ──▶│ LSTM │      │
│   │ cell │     │ cell │           │ cell │      │
│   └──────┘     └──────┘           └──┬───┘      │
│      ↑              ↑                │           │
│  [br₁,t₁]      [br₂,t₂]         last           │
│                               hidden state       │
└─────────────────────────────────────────────────┘
         │
         ▼  (last hidden state hₙ)
┌─────────────────────────────────────────────────┐
│         Fully Connected Layer (Linear)           │
│              hidden_size → 1                     │
└─────────────────────────────────────────────────┘
         │
         ▼
  Number of planets (f64 → rounded to integer)
```

**Why LSTM and not a simple RNN or CNN?**

LSTM solves the vanishing gradient problem and is capable of "remembering" patterns that are spread across time — in this case, periodic transits that may occur once every few weeks or months. CNNs are good at detecting local patterns but handle long-range dependencies less effectively.

**Key model parameters:**

| Parameter | Value | Description |
|----------|----------|----------|
| `input_size` | 2 | Brightness + Time per step |
| `output_size` | 1 | Predicted number of planets |
| Loss function | MSE | Mean Squared Error |
| Optimizer | Adam | Adaptive optimization |
| Progress report | every 2.5% of epochs | Outputs current loss |

---

## Project structure

```
PlanetFinder/
│
├── src/                    # Rust source code
│   └── main.rs             # Entry point and CLI
│   └── data.rs             # Utility for reading learn files
│   └── ai.rs               # Neural network training and prediction
│   └── web.rs              # Web server
│
├── download_data.py        # Python script for downloading NASA data
│
├── Cargo.toml              # Rust package dependencies and metadata
├── .gitignore              # Git exclusions
│
├── model.ot                # Saved model (created after training)
├── learn1.txt              # Training file 1 (created by download_data.py)
├── learn2.txt              # Training file 2
├── ...                     # learnN.txt — as many as you download
│
├── README.md               # This file
└── LICENSE                 # Apache 2.0
└── PitchDeck.pdf           # Presentation
```

---

## Code overview

### `src/main.rs` — Entry point and CLI

Implements an interactive text interface. The user selects the operating mode via arguments.
`train` — training, `predict` — prediction, `web <port>` — starting the web server.

### `src/data.rs` — Utility for reading learn files

Reads and returns training file data in a convenient format for the program.

### `src/ai.rs` — Neural network training and prediction

In **training** mode, the program:
1. Scans the directory for `learn*.txt` files
2. Parses each file: reads `(brightness, time)` pairs and the `result N` label
3. Normalizes brightness (by dividing by the mean) and time (to the range `[0, 1]`)
4. Builds tensors and starts the training loop
5. Prints progress and the current MSE loss every 2.5% of epochs
6. Automatically saves `model.ot` when the minimum loss is achieved

In **prediction** mode, the program:
1. Loads `model.ot` from disk
2. Accepts `(brightness, time)` pairs from standard input until the `end` command
3. Runs the sequence through the LSTM and outputs the prediction

### `src/web.rs` — Web server

In **web server** mode, the program:
1. Loads `model.ot` from disk
2. Starts a server for a website accessible at `http://localhost:{specified port}`

### `download_data.py` — Downloading NASA data

A Python script using the `lightkurve` library to access Kepler and TESS archives. It downloads light curves of stars with a known number of planets and converts them to a format understood by PlanetFinder.

### `Cargo.toml` — Rust dependencies

One of the project's key dependencies is `tch` (Rust bindings to LibTorch, the C++ API of PyTorch). It is used to implement tensor operations, the LSTM layer, and saving/loading the model in `.ot` format.
The web server is built on top of the Rust framework `actix-web`.

---

## Data format

Each training file is named `learn1.txt`, `learn2.txt`, ... and has the following structure:

```
0.998 131.2
1.002 132.1
0.995 133.0
0.999 133.9
0.872 134.8    ← transit: brightness dropped
0.870 135.7    ← transit: continues
0.998 136.6
...
result 2       ← label: 2 planets in the system
```

| Field | Type | Description |
|------|-----|----------|
| `brightness` | `f64` | Normalized stellar brightness (≈ 1.0 under normal conditions) |
| `time` | `f64` | Observation timestamp (BJD or arbitrary units) |
| `result N` | string | Last line — number of known planets |

> **Important:** files must be named strictly `learnN.txt` (e.g., `learn1.txt`, `learn42.txt`). The program finds them automatically using the pattern `learn*.txt`.

---

## Requirements

> You can also try the [neural network online](https://planetfinder.online), without a local installation.

### Required

| Component | Version | Where to get |
|-----------|--------|-----------|
| **Rust** | 1.70+ | [rustup.rs](https://rustup.rs/) |
| **LibTorch** | 2.x | [pytorch.org](https://pytorch.org/get-started/locally/) |

### Optional

| Component | Purpose | Where to get |
|-----------|-----------|-----------|
| **CUDA** 11.x+ | GPU-accelerated training | [developer.nvidia.com](https://developer.nvidia.com/cuda-toolkit) |
| **Python 3.8+** | Only for `download_data.py` | [python.org](https://python.org) |
| **lightkurve, astropy, numpy** | Downloading NASA data | `pip3 install lightkurve astropy numpy` |

### LibTorch environment variable

```bash
# Linux / macOS
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Windows (PowerShell)
$env:LIBTORCH = "C:\path\to\libtorch"
$env:Path = "$env:LIBTORCH\lib;$env:Path"
```

---

## Installation

### Step 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version   # should output: rustc 1.70.0 or higher
```

### Step 2. Download LibTorch

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select:
- **Package:** LibTorch
- **Language:** C++
- **OS:** your OS
- **CUDA:** the version you need (or CPU-only)

Extract the archive and set the `LIBTORCH` variable (see above).

### Step 3. Clone the repository

```bash
git clone https://github.com/Ztry8/PlanetFinder.git
cd PlanetFinder
```

### Step 4. Build the project

```bash
cargo build --release
```

> The first build takes several minutes due to dependency compilation. Subsequent builds are significantly faster. The compiled binary will appear in `target/release/`.

---

## Downloading NASA data

### Via the script (recommended)

```bash
pip3 install lightkurve astropy numpy
python3 download_data.py
```

The script will download light curves from the Kepler/TESS archive and save them as `learn1.txt`, `learn2.txt`, ... in the working directory.

### Pre-built data from the release

If you don't have Python access or want to get started immediately — download the ready-made `.txt` files and the pre-trained `model.ot`:

**[Download ready-made files → Releases v1.0.1](https://github.com/Ztry8/PlanetFinder/releases/tag/v1.0.1)**

Place the downloaded files in the project's root directory.

---

## Usage

```bash
cargo run --release <mode>
```

### Mode 1: Training the model

```
Found training files: 42
Starting training...

Completed   50 epochs ( 2.5% done) — current error: 1.2885
Completed  100 epochs ( 5.0% done) — current error: 0.9341
Completed  150 epochs ( 7.5% done) — current error: 0.7102
...
Completed 2000 epochs (100.0% done) — current error: 0.0487

Best model saved: model.ot
```

The program automatically finds all `learn*.txt` files, trains the LSTM, and saves the best weights to `model.ot`.

### Mode 2: Prediction

```
Model loaded: model.ot
Enter "brightness time" pairs, one per line.
Enter "end" to get the result.

> 0.998 131.2
> 1.002 132.1
> 0.872 134.8
> 0.870 135.7
> 0.999 136.6
> end

Predicted number of planets: 1
```

> For prediction, the `model.ot` file must be in the working directory. Either train the model (mode 1) or download the pre-trained one from the release.

---

## Training file examples

### Hot Jupiter (1 planet, deep transit)

```
# Large planet, orbit ~3 days
1.001 0.0
1.000 0.5
0.830 1.0    ← transit: -17% brightness
0.829 1.5
0.831 2.0
1.000 2.5
0.829 4.0    ← repeated transit, same period
result 1
```

### Multi-planet system (2 planets)

```
# Small transit (period ~5 days) + large transit (period ~18 days)
1.000 0.0
0.991 5.3    ← small planet (-0.9%)
1.001 10.6
0.992 15.9
0.870 18.2   ← large planet (-13%)
1.000 21.2
0.991 26.5
result 2
```

---

## Performance

| Mode | Configuration | Speed |
|-------|-------------|---------|
| Training | CPU (Intel i7) | ~200 epochs/sec |
| Training | GPU (RTX 3060, CUDA) | ~2000 epochs/sec |
| Prediction | CPU or GPU | < 10 ms |

Accuracy depends on the number of training files (50+ recommended), their diversity, and the number of epochs.

---

## FAQ

**Q: Can I use my own data, not from NASA?**
A: Yes. Any data in the format `brightness time` line by line, with a closing line `result N`, will be accepted by the program.

**Q: Why is normalization necessary?**
A: Different stars have different baseline brightness levels. Normalization brings all curves to a common scale, allowing the model to learn from transit patterns rather than absolute values.

**Q: How long does the model need to train?**
A: For 50 files on CPU — approximately 15–30 minutes at 2000 epochs. With GPU — roughly 10 times faster.

**Q: Does the project work on Apple Silicon (M1/M2/M3)?**
A: Yes, in CPU mode. Metal/MPS support depends on the LibTorch version.

**Q: Why Rust and not Python?**
A: Rust provides high execution speed, memory safety without a GC, and the ability to compile into a single binary without dependencies on an interpreter.

---

## Sources and references

### Data

| Source | Description | Link |
|---------|----------|--------|
| **NASA Kepler** | Primary source of light curves | [nasa.gov/kepler](https://www.nasa.gov/mission_pages/kepler/main/index.html) |
| **NASA TESS** | Extended survey of transiting exoplanets | [tess.mit.edu](https://tess.mit.edu/) |
| **NASA Exoplanet Archive** | Catalog of confirmed exoplanets | [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu/) |

### Libraries

| Library | Language | Purpose | Link |
|-----------|------|-----------|--------|
| `tch-rs` | Rust | LibTorch (PyTorch C++) bindings for LSTM and tensors | [github.com/LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs) |
| `actix-web` | Rust | Framework for building the web server | [actix.rs](https://actix.rs/) |
| `lightkurve` | Python | Downloading and processing Kepler/TESS light curves | [docs.lightkurve.org](https://docs.lightkurve.org/) |
| `astropy` | Python | Astronomical calculations and formats | [astropy.org](https://www.astropy.org/) |
| `numpy` | Python | Numerical operations | [numpy.org](https://numpy.org/) |

---

## License

Distributed under the **Apache License 2.0** — free to use, including commercially, provided attribution is retained. See the [LICENSE](LICENSE) file for details.

---