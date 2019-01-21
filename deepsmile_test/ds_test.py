# Test the functionality of the deepsmile representation from https://github.com/nextmovesoftware/deepsmiles

import deepsmiles
from rdkit import Chem
import pandas as pd
import numpy as np


converter = deepsmiles.Converter(rings=True, branches=True)


def can_smile(smi_list):
    """
    Generate standard SMILES from input
    Args:
        smi_list: list of SMILES

    Returns: canolized smile list

    """
    can_list = []
    for item in smi_list:
        if Chem.MolFromSmiles(item) is not None:
            can_item = Chem.MolToSmiles(Chem.MolFromSmiles(item))
            can_list.append(can_item)
    return can_list


def convert_file(smi_file):
    """
    Analyse the performance of deepsmile for the input file
    Args:
        smi_file: path of input file containing SMILE strings

    Returns:
        percentage of successfully converted smiles
        file containing deepsmile representations

    """
    out = open('deep'+smi_file, 'w')
    try:
        smi_f = pd.read_csv(smi_file, header=None)
    except FileNotFoundError:
        print('Input file not found. Please make sure file path is correct.')
    smi_lst = smi_f[0].tolist()

    deep_lst = [converter.encode(smi) for smi in smi_lst]
    decoded_lst = []
    num_decode, num_recover= 0, 0
    for i in range(len(deep_lst)):
        try:
            decoded = converter.decode(deep_lst[i])

        except deepsmiles.DecodeError as e:
            decoded = None
            print("DecodeError! Error message was {}".format(e.message))
        decoded_lst.append(decoded)
        if decoded:
            num_decode += 1
    smi_can_lst = can_smile(smi_lst)
    decoded_can_lst = can_smile(decoded_lst)
    num_recover = sum(smi_can_lst[i] == decoded_can_lst[i] for i in range(len(smi_can_lst)))
    for i in range(len(smi_lst)):
        if smi_can_lst[i] == decoded_can_lst[i]:
            out.write(deep_lst[i]+'\n')
    out.close()


    return num_decode, num_recover, num_decode*100/len(smi_lst), num_recover*100/len(smi_lst)


if __name__ == '__main__':
    num,rnum, percent,rpercent= convert_file(input())
    print('{} SMILES successfully decoded, which is {}% of the input SMILES.'.format(num, percent))
    print('{} SMILES successfully recovered, which is {}% of the input SMILES.'.format(rnum, rpercent))