# Class Activation Map (CAM) using CNN on CIFAR-10

This repository contains a project that demonstrates how to generate Class Activation Maps (CAM) using a Convolutional Neural Network (CNN) built with the Keras API and trained on the CIFAR-10 dataset.

## Project Structure

- `data/`: Directory for dataset storage.
- `models/`: Contains the model architecture.
- `notebooks/`: Jupyter notebooks for experiments and visualizations.
- `scripts/`: Python scripts for training and CAM generation.
- `requirements.txt`: List of required packages.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/class-activation-map.git
   cd class-activation-map
2. Install the required package
   ```bash
   pip install -r requirements.txt
   
## Usage

### Training the model:
To train the model, run:
```bash
python scripts/train.py
```

### Generating Class Activation Maps
To generate CAMs, run:
```bash
python scripts/cam.py
```

