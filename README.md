# **Dissolved Oxygen Estimation Device**

**By:** Gison J.J., Hidalgo J.M., Yanson K.

**Class:** BS ECE 4A

**Technological University of the Philippines Visayas**

## Project Overview

A low-cost, floating device that estimates dissolved oxygen (DO) levels in aquaculture ponds using machine learning. It uses five water quality sensors (temperature, pH, turbidity, conductivity, and atmospheric pressure) and processes data via Raspberry Pi 4. A stacked ML model achieved 3.21% mean error, closely matching lab-grade DO meters.

## Features

* Real-time DO estimation
* Trained ML models: RF, SVR, XGB, Stacked
* Sensors: Temp, pH, Turbidity, Conductivity, Barometric Pressure
* Floating PVC platform with SD card logging
* Kodular-based mobile UI
* Verified accuracy via t-tests and SUS scoring

## Tech Stack

* **Hardware:** Raspberry Pi 4, analog/digital sensors, SD module
* **Software:** Python, scikit-learn, XGBoost, Kodular
* **Models:** RF, SVR, XGB, Stacked Model

## Installation

1. Connect sensors to Raspberry Pi 4
2. Upload `.pkl` ML model files
3. Install Python libs:
   `pip install pandas scikit-learn xgboost`
4. Run the Python script
5. Use Kodular app for mobile display

## Usage

* Deploy device in pond
* Sensors collect data, Pi estimates DO
* Results are saved and sent via Bluetooth

## Contributors

* Gison J.J.
* Hidalgo J.M.
* Yanson K.

## License

MIT License
