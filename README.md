# MNIST Digit Classification with PyTorch Lightning

A modular convolutional neural network (CNN) implementation for digit classification using the MNIST dataset, built with PyTorch and PyTorch Lightning.

## Prerequisites
* [Conda](https://docs.conda.io/)

## Project Structure
```text
├── model.py           # Model definitions
├── main.ipynb         # Training, evaluation, and visualization
├── environment.yml    # Conda environment
├── .gitignore         # Excluded files
└── README.md
```

## Setup
```text
git clone https://github.com/akshayvmehta/pytorch-mnist-classifier
cd pytorch-mnist-classifier
conda env create -f environment.yml
conda activate mnist_pl
```

## Highlights

Modular Design: Separation of logic (model.py) from execution/interface (main.ipynb).

Visualization: Full documentation in the notebook, including training loss/accuracy curves and inference visualisations.

Dynamic Architecture: Uses a "Dummy Pass" in __init__ for automatic input size calculation, making the model architecture flexible.

## Results
The model achieves high accuracy on the MNIST validation set. Final metrics and visual plots are available directly in main.ipynb.

Created as a learning project for PyTorch Lightning and GitHub workflow.
