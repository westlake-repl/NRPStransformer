[![Website Status Check](https://github.com/westlake-repl/NRPSTransformer/actions/workflows/http_check.yml/badge.svg)](https://github.com/westlake-repl/NRPSTransformer/actions/workflows/http_check.yml)
[![DOI](https://zenodo.org/badge/961756607.svg)](https://doi.org/10.5281/zenodo.15773640)

# NRPSTransformer
Transformer Based Nonribosomal Peptide Synthetases Predictor.

Welcome to visit and try our web demo http://www.nrpstransformer.cn

## Install [Might Take 5-10 min]
First clone our repo by
```bash
git clone https://github.com/westlake-repl/NRPSTransformer.git
cd NRPSTranformer
```
Then you can set up a conda virtual environmenmt for this project
```bash
conda create -n NRPS python==3.8.5
pip install -r requirement.txt
conda activate NRPS
```
After that, install necessary backbone model
```bash
git lfs install
git clone https://huggingface.co/facebook/esm2_t33_650M_UR50D
```
Then install our checkpoint for inference


## Inference
```bash
./run.sh
```

After the program prompted:
