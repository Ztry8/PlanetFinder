# Planet Finder
[![GitHub last commit](https://img.shields.io/github/last-commit/ztry8/planetfinder)](https://github.com/ztry8/planetfinder)
[![License](https://img.shields.io/github/license/ztry8/planetfinder)](https://github.com/ztry8/planetfinder/blob/main/LICENSE)

## Predict the number of exoplanets from lightcurve data using Rust and LSTM

### About
This project is a **lightweight Rust program** that uses an LSTM neural network to predict the number of exoplanets around stars based on **lightcurve observations**. 

The program supports **training** from `.txt` datasets and **interactive prediction** from user input.

### Why It May Be Useful

#### Astronomy Perspective
- **Exoplanet Detection**: Helps astronomers estimate the number of exoplanets around stars using lightcurve data.  
- **Data Analysis Efficiency**: Automates analysis of large datasets, saving significant time compared to manual methods.  
- **Exploration and Research**: Enables testing hypotheses about star systems and transit patterns quickly.  

#### Programming / IT Perspective
- **Machine Learning Practice**: Provides a hands-on example of LSTM neural networks for time-series prediction.  
- **Customizable and Extensible**: Users can train the model on their own datasets or adjust parameters for experimentation.  
- **Cross-Platform & GPU Support**: Runs on Linux, Windows, and macOS, with optional CUDA acceleration for faster training.  
- **Interactive Usage**: Allows real-time input of data and instant predictions, making it suitable for demos and prototyping.

### Features
- LSTM-based sequence model for time-series data
- Handles variable-length lightcurves automatically
- Interactive prediction mode
- Normalization of flux and time for stable training
- Optional GPU acceleration via CUDA
- Progress logging every 2.5% of epochs
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

### Downloading NASA Data

The project uses datasets of exoplanet observations provided by NASA.    
You can download the necessary .txt files using the included Python script `download_data.py`.    
You can also download ready-made files from [this page](https://github.com/Ztry8/PlanetFinder/releases/tag/v1.0.1).

#### Installing dependencies

Make sure you have Python 3 installed. Then install required packages:

```
pip3 install lightkurve astropy numpy
```

#### Usage

```
python3 download_data.py
```

The script will automatically download the latest datasets into your working folder.    
Each file will be named in the format `learn_<number>.txt.`

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
You can also download ready-made model from [this page](https://github.com/Ztry8/PlanetFinder/releases/tag/v1.0.1).

Run the program:
```bash
cargo run --release
```

- Enter `1` to start training.  
- The program will scan for `learn*.txt` files.  
- Training progress is displayed every 2.5% of total epochs:
```
Completed   50 epochs (2.5 % done) - current error: 1.2885
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
