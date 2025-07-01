[![Website Status Check](https://github.com/westlake-repl/NRPSTransformer/actions/workflows/http_check.yml/badge.svg)](https://github.com/westlake-repl/NRPSTransformer/actions/workflows/http_check.yml)
[![DOI](https://zenodo.org/badge/961756607.svg)](https://doi.org/10.5281/zenodo.15773640)

# NRPSTransformer
**A Transformer-Based Predictor for Nonribosomal Peptide Synthetases (NRPS) Specificity**

Welcome! You can try our web demo here: http://www.nrpstransformer.cn

## Overview
NRPSTransformer is a deep learning model designed to predict the substrate specificity of adenylation (A) domains in Nonribosomal Peptide Synthetases (NRPS). It employs a transformer-based architecture that capitalizes on transfer learning. It generates high-quality protein embeddings from a large, pre-trained language model, which are subsequently fed into a fine-tuned classifier to achieve state-of-the-art prediction accuracy of 93%.

## Prerequisites
Before you begin, ensure you have the following installed:

- **Conda/Miniconda**: To manage the project environment. You can find installation instructions here.
- **Git LFS**: For handling large model files. You can install it by following the instructions here.

## Installation [Estimated Time: 10-20 min]
Please follow these steps carefully to set up the project environment and download the necessary models.

**1. Clone the Repository**
First, clone this repository to your local machine.

 ```bash
git clone https://github.com/westlake-repl/NRPSTransformer.git
cd NRPSTransformer
 ```

**2. Set Up the Conda Environment**
Create and activate a dedicated conda environment using the provided requirement.txt file.

```bash
# Create a new conda environment named 'NRPS' with Python 3.8.5
conda create -n NRPS python=3.8.5 -y
# Activate the newly created environment
conda activate NRPS
# Install the required Python packages
pip install -r requirement.txt
```

**3. Download the ESM-2 Backbone Model**
This project uses the esm2_t33_650M_UR50D model as a backbone. You need Git LFS to clone it from Hugging Face.
```bash
# Initialize Git LFS
git lfs install
# Clone the ESM-2 model repository
git clone https://huggingface.co/facebook/esm2_t33_650M_UR50D
```

**4. Download Project Checkpoints and Labels**
The model checkpoints and class labels are hosted on Zenodo. Download them and place them in the correct directories.
```bash
# Create the necessary directories
mkdir -p checkpoints
mkdir -p model/class_label

# Download checkpoint files into the 'checkpoints' directory
wget -O checkpoints/all.ckpt https://zenodo.org/records/15771488/files/all.ckpt
wget -O checkpoints/benchmark.ckpt https://zenodo.org/records/15771488/files/benchmark.ckpt
wget -O checkpoints/clade.ckpt https://zenodo.org/records/15771488/files/clade.ckpt

# Download label mapping files into the 'model/class_label' directory
wget -O model/class_label/labelid2lable-17.pt https://zenodo.org/records/15771488/files/labelid2lable-17.pt
wget -O model/class_label/labelid2label-43.pt https://zenodo.org/records/15771488/files/labelid2label-43.pt
```

After completing these steps, your project directory should look like this:

```bash
NRPSTransformer/
├── checkpoints/
│   ├── all.ckpt
│   ├── benchmark.ckpt
│   └── clade.ckpt
├── esm2_t33_650M_UR50D/
│   └── ... (Hugging Face model files)
├── model/
│   └── class_label/
│       ├── labelid2lable-17.pt
│       └── labelid2label-43.pt
├── requirement.txt
├── run.sh
└── ... (other project files)
```

## Usage
The primary way to run predictions is through the run.sh script. It provides an interactive command-line interface to specify your input and output files.

**1. Execute the Script**
Make sure you are in the NRPSTransformer root directory and the NRPS conda environment is active.
```bash
bash run.sh
```

**2. Interactive Prompts**
The script will prompt you for two pieces of information:

- **Input File Path**: The path to your FASTA file containing the protein sequences of the A-domains you want to predict.
- **Result File Path**: The path where the prediction results will be saved in CSV format.
You can either type a new path or press Enter to accept the default values shown in the parentheses.

Here is an example of the terminal interaction:

```bash
Please enter the input sequence file path (Default: ./sequence/sequence.fasta):
# User can type a path like './my_sequences.fasta' or just press Enter

setting input file to ./sequence/sequence.fasta

Please enter the result file path (Default: ./result/result.csv): 
# User can type a path like './my_results.csv' or just press Enter

setting result path to ./result/result.csv

# The prediction process will start now...
# ...
# ...
# Prediction finished. Results are saved in ./result/result.csv
```

The program will then run the prediction pipeline, and upon completion, a message will confirm that the results have been saved to your specified output file.
