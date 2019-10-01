# RNN and RL model for molecular generation
Models and codes for the paper: 
[Molecular Generation Targeting Desired Electronic Properties via Deep Generative Models](https://chemrxiv.org/articles/Molecular_Generation_Targeting_Desired_Electronic_Properties_via_Deep_Generative_Models/9913865)

Prior model adapted and modified from https://arxiv.org/abs/1704.07555
## Requirements

Python 3.6

PyTorch 0.1.12

RDkit

Scikit-Learn (for QSAR scoring function)

tqdm (for training Prior)

## Usage
To train a Prior model starting with a SMILES file called mols.smi:

First filter the SMILES and construct a vocabulary from the remaining sequences. ./data_structs.py mols.smi - Will generate data/CEP_cano.smi and data/Voc_cep.
A filtered file containing around 1.1 million SMILES from the Guacamol is provided in ChEMBL_from_gua_filter.smi.

Then use ./train_prior.py to train the Prior. A pretrained Prior is included.

To do transfer learning on a target dataset, use transfer_userinpt.py.
