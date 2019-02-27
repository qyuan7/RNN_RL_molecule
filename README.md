# RNN and RL model for molecular generation
Prior model adapted and modified from https://arxiv.org/abs/1704.07555
## Requirements

Python 3.6

PyTorch 0.1.12

RDkit

Scikit-Learn (for QSAR scoring function)

tqdm (for training Prior)

## Usage
To train a Prior starting with a SMILES file called mols.smi:

First filter the SMILES and construct a vocabulary from the remaining sequences. ./data_structs.py mols.smi - Will generate data/mols_filtered.smi and data/Voc. A filtered file containing around 1.1 million SMILES and the corresponding Voc is contained in "data".

Then use ./train_prior.py to train the Prior. A pretrained Prior is included.

To train an Agent using our Prior, use the main.py script. For example:

./main.py --scoring-function activity_model --num-steps 1000
