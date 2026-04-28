MNIST Digit Classification with PyTorch Lightning

A modular implementation of a Convolutional Neural Network (CNN) for digit classification using the MNIST dataset. This project serves as a structured introduction to building deep learning models with PyTorch and PyTorch Lightning.

Project Structure

Plaintext
├── model.py           # Contains the MNISTCNN and MNISTClassifier classes
├── main.ipynb         # Jupyter Notebook for training, evaluation, and visualization
├── environment.yml    # Conda environment configuration
├── .gitignore         # Excludes logs and data folders from version control
└── README.md

Setup and Installation

To run this project, ensure you have Conda installed.

Clone the repository:

Bash
git clone https://github.com/akshayvmehta/pytorch-mnist-classifier
cd pytorch-mnist-classifier

Create the environment:

Bash
conda env create -f environment.yml
conda activate mnist_pl

Project Highlights

Modular Design: Logic is separated into model.py for clean code maintenance, while main.ipynb acts as the primary interface.

Visualization: The main notebook is fully documented with outputs, including:

Loss and Accuracy Curves: Track the training progress over epochs.

Inference Examples: Direct visualization of model predictions against true labels.

Dynamic Architecture: Uses a "Dummy Pass" approach in the __init__ method, allowing for flexible modifications to the CNN layers without needing to manually recalculate input sizes for linear layers.

Results

The model achieves high accuracy on the MNIST validation set. You can view the final metrics and the visual plots directly in main.ipynb.

Created as a learning project for PyTorch Lightning and GitHub workflow.