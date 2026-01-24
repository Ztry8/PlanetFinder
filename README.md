# Planet Finder
[![GitHub last commit](https://img.shields.io/github/last-commit/ztry8/planetfinder)](https://github.com/ztry8/planetfinder)
[![License](https://img.shields.io/github/license/ztry8/planetfinder)](https://github.com/ztry8/planetfinder/blob/main/LICENSE)

## Predict the number of exoplanets from lightcurve data using Rust and LSTM

### About
This project is a **lightweight Rust program** that uses an LSTM neural network to predict the number of exoplanets around stars based on **lightcurve observations**. 

The program supports **training** from `.txt` datasets and **interactive prediction** from user input.

### Features
- LSTM-based sequence model for time-series data
- Handles variable-length lightcurves automatically
- Interactive prediction mode
- Normalization of flux and time for stable training
- Optional GPU acceleration via CUDA
- Progress logging every 10% of epochs
- Saves the best model automatically

### Data Format
Files must be named like `learn1.txt`, `learn2.txt`, etc.  
Each file contains:

```
0.998 131.2
1.002 132.1
...
result 2
```

Where each line is `flux time` and the last line indicates the number of planets with `result N`.

### Installation

#### Requirements
- Rust 1.70+  
- Optional: CUDA GPU for faster training  

#### Build
```bash
git clone https://github.com/Ztry8/PlanetFinder.git
cd PlanetFinder
cargo build --release
```

The program is cross-platform and runs on Linux, Windows, and macOS.

### Usage

#### Training
Run the program:
```bash
cargo run --release
```

- Enter `1` to start training.  
- The program will scan for `learn*.txt` files.  
- Training progress is displayed every 10% of total epochs:
```
Completed 50 epochs (10% done) - current error: 48.0%
```
- The best model is saved automatically as `model.ot`.

#### Prediction
Run the program:
```bash
cargo run --release
```

- Enter `2` to predict.  
- Input `flux time` pairs line by line.  
- Type `end` to finish input.  
- The program outputs the predicted number of planets:
```
Predicted number of planets: 3
```

### Downloading NASA Data

The project uses datasets of exoplanet observations provided by NASA.    
You can download the necessary .txt files using the included Python script `download_data.py`.

#### Installing dependencies

Make sure you have Python 3 installed. Then install required packages:

```
pip3 install lightkurve astropy numpy
```

Using the script:
```
python3 download_data.py
```

The script will automatically download the latest datasets into your working folder.    
Each file will be named in the format `learn_<number>.txt.`
