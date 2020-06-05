#!usr/bin/env python
"""
Evaluate the performance of the generative model on multiple aspects:
to be filled
"""

import pandas as pd
import numpy as np
from post_processing import data
from rdkit import Chem, DataStructs
import scipy.stats as ss
import math
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
import time
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def internal_sim(smi_lst):
    """
    Compute internal similarity within generated SMILES
    Args:
        smi_lst: list of generated unique SMILE structures

    Returns: Average internal molecular similarity with in the input list

    """
    setV =  len(smi_lst)
    mols = data.get_mols(smi_lst)
    fps_morgan, _ = data.get_fingerprints(mols)
    total_smi = 0
    for i in range(len(fps_morgan)):
        for j in range(len(fps_morgan)):
            total_smi += DataStructs.DiceSimilarity(fps_morgan[i], fps_morgan[j])
    Din = total_smi/(setV*setV)
    return Din


def external_sim(smi_lst, reference):
    """
    Compute the external similarity against the source data, i.e. the average similarity between the
    generated molecules and their nearest neighbours in the training set.
    Args:
        smi_lst: list of generated unique SMILE structures
        reference: list of SMILES used for training the generation

    Returns: Average external molecular similarity between generated and origin lst

    """
    gen_mols, ori_mols = get_mols(smi_lst), get_mols(reference)
    fps_gen, _ = get_fingerprints(gen_mols)
    #print(len(smi_lst), len(fps_gen))
    fps_ori, _ = get_fingerprints(ori_mols)
    similarity_maxs = []
    neighbours = []
    for i in range(len(fps_gen)):
        similarity_with = []
        gen_smi =  smi_lst[i]
        for j in range(len(fps_ori)):
            similarity_with.append(DataStructs.DiceSimilarity(fps_gen[i], fps_ori[j]))
        similarity_maxs.append(max(similarity_with))
        k = np.argmax(similarity_with)
        ref_neighbour = similarity_with[k]
        neighbours.extend([reference[k], ref_neighbour])
    assert (len(similarity_maxs) == len(fps_gen))
    Dext = np.sum(similarity_maxs)/len(fps_gen)
    return Dext, neighbours


def KL_divergence(gen_arr, reference_arr):
    """

    Args:
        gen_arr: array of numeric parameters of generated molecules
        reference_arr: array of original numeric parameters of training molecules

    Returns: KL-divergence of value_arr against reference_arr

    """
    epsilon = 0.0001
    min_val = math.floor(min(min(gen_arr), min(reference_arr)))
    max_val = math.ceil(max(max(gen_arr), max(reference_arr)))
    gen_arr_dis = np.histogram(gen_arr, bins=12,range=(min_val, max_val), density=True)[0] + epsilon
    reference_arr_dis = np.histogram(reference_arr, bins=12, range=(min_val, max_val), density=True)[0] + epsilon
    entropy = ss.entropy(reference_arr_dis, gen_arr_dis)
    return entropy


def generate_metric_df():
    all_exp_df = pd.read_csv('exp_df_merged.csv')
    all_gen_df = pd.read_csv('novel_sampled_merged.csv')
    eval_df = pd.DataFrame()
    eval_df['Group'] = ['all', 'class3', 'prom']
    internal_sims = []; external_sims = []; gaps_kls = []; dips_kls= []
    for group in ['all', 'class3', 'prom']:
        gen_smi = all_gen_df[all_gen_df['Label'] == group]['SMILES'].tolist()
        exp_smi = all_exp_df[all_exp_df['Label'] == group]['SMILES'].tolist()
        gen_gap = all_gen_df[all_gen_df['Label'] == group]['Gaps']
        exp_gap = all_exp_df[all_exp_df['Label'] == group]['gaps']
        gen_dip = all_gen_df[all_gen_df['Label'] == group]['Dips']
        exp_dip = all_exp_df[all_exp_df['Label'] == group]['dips']
        internal_ = internal_sim(gen_smi)
        internal_sims.append(internal_)
        external_ , _= external_sim(gen_smi, exp_smi)
        external_sims.append(external_)
        gaps_kl = KL_divergence(gen_gap, exp_gap)
        dips_kl = KL_divergence(gen_dip, exp_dip)
        gaps_kls.append(gaps_kl)
        dips_kls.append(dips_kl)
        print('Internal similarity for group {}: {}'.format(group, internal_))
        print('External similarity for group {}: {}'.format(group, external_))
        print('KL divergence for H-L gaps for group {}: {}'.format(group, gaps_kl))
        print('KL divergence for dips for group {}: {}'.format(group, dips_kl))
    eval_df['Internal_similarity'] = internal_sims
    eval_df['External_similarity'] = external_sims
    eval_df['KL_gaps'] = gaps_kls
    eval_df['KL_dips'] = dips_kls
    return eval_df


def find_neighbour(smi, b_lst, n=5):
    """
    get n neighbours (most similar molecules) of smi from b_lst
    IMPORTANT: all smiles must be valid.
    Args:
        smi: target smile representation of molecule
        b_lst: list of smiles
        n: number of neighbours to obtain

    Returns: list of smiles of the n neighbours

    """
    smi_mol, lst_mols = get_mols([smi]), get_mols(b_lst)
    fps_lst, _ = get_fingerprints(lst_mols)
    smi_fp, _ = get_fingerprints(smi_mol)
    assert len(fps_lst) == len(b_lst), "Invalid SMILES representation present."
    similarity = []
    for i in range(len(fps_lst)):
        tmp_sim = DataStructs.DiceSimilarity(smi_fp[0], fps_lst[i])
        similarity.append((b_lst[i], tmp_sim))
    sorted_sim = sorted(similarity, key=lambda tup:tup[0])
    return sorted_sim[:n]


def moltosvg(mol,molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')


def get_prom_neighbours():
    """
    Get the closest neighbours of gen prom molecules in the reference set
    Returns:

    """
    all_exp_df = pd.read_csv('exp_df_merged.csv')
    all_gen_df = pd.read_csv('novel_sampled_merged.csv')
    exp_prom = all_exp_df[all_exp_df['Gaps'] <=2 & all_exp_df['Dips']<=2]


if __name__ == '__main__':
    m1_train = pd.read_csv('Training')





