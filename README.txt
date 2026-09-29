Event Classification with KAN Networks
This repository contains the data and source code for the training, validation, and symbolic interpretability of an electrical fault diagnosis system using Kolmogorov-Arnold (KAN) Neural Networks and Graph Signal Processing (GSP).

The data and scripts correspond to the evaluation of a reduced equivalent model of the 195-bus Mexican Interconnected System.
Repository Contents
Dataset: The test data is located in the following link: https://www.dropbox.com/scl/fo/xsmhz64kbmuizpfloclwx/AFQBZI4A2WM3LeRNTCkwmoo?rlkey=ooh934h990z0bepvob8hxk8hi&dl=0
kan_master.py: Main script that performs graph processing (Graphical Lasso), spectral feature extraction, and multi-class training of the KAN network, culminating in the extraction of equations via symbolic regression.
kan_interpretability.py: Validation script that mathematically evaluates the extracted algebraic equations against the test set. About the Dataset (PST - MATLAB)
The raw data were generated using dynamic simulations in MATLAB’s Power System Toolbox (PST).

The results are saved in .mat files containing a structure named sstr. You can use any other dataset using PST.
The variables extracted from the sstr structure for this analysis are the voltages (bus_v) and frequencies (bus_freq), along with their time vector (t).
Folder Structure: Each event type or class (e.g., Generator Trip, Line Trip, Load Trip) has a folder that groups all the .mat files containing the sstr structure corresponding to those events.
Usage and Configuration Instructions
Before running the script, make sure your local folders are structured as follows:

├── data/              # Subfolders by event type containing .mat (sstr) files

├── checkpoints/       # Will be created automatically to store tensors (.pt) and equations (.txt)

├── figures/           # Will be created automatically to store plots (.pdf)

├── kan_master.py

└── kan_interpretability.py
Step 1: Training and Feature Extraction (kan_master.py)
 (./data, ./checkpoints). Make sure the data/ folder contains your .mat simulations. When you run this script, the code will automatically read the sstr structure, construct the graphs, train the multi-class KAN model, and generate the boundary_equations.txt file with the extracted formulas.

python kan_master.py
Step 2: Interpretability and Metrics (kan_interpretability.py)
This script reads directly from the processed dataset (dataset_tensors_mc_physics.pt) and the formulas in boundary_equations.txt.
python kan_interpretability.py