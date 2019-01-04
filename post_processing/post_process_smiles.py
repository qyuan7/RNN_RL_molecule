import pandas as pd
import numpy as np
from copy import deepcopy
from rdkit import Chem
from data import *
from sklearn.externals import joblib
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot


def get_smi_list_overlap(large, small):
    """

    Args:
        large: list containing the SMILE structures for transfer training
        small: list containing the SMILE structures for transfer sampling

    Returns: num of repeat SMILES, num of unique SMILES in transfer sampling, list of unique SMILES

    """
    def can_smile(smi_list):
        can_list = []
        for item in smi_list:
            if Chem.MolFromSmiles(item) is not None:
                can_item = Chem.MolToSmiles(Chem.MolFromSmiles(item))
                can_list.append(can_item)
        return can_list
    large_can, small_can = can_smile(large), can_smile(small)
    small_copy = deepcopy(small_can)
    overlap = set(small_can).intersection(large_can)
    for item in overlap:
        small_copy.remove(item)
    return len(overlap), len(small_copy), small_copy


def predict_property(model_file, fps):
    """
    Function to predict the properties of generated molecules
    Args:
        model_file: File containing pre-trained ML model for prediction
        fps: list of molecular fingerprints

    Returns: list of predicted valued

    """
    model = joblib.load(model_file)
    return model.predict(fps)



def save_predict_results():
    """
    Predict the gap and dip of generated SMILES from files and save the results
    Also save the generated with gap < 2, dip <2 as promising candidates
    Returns:

    """

    ori_df = pd.read_csv('./sampled_da_info/refined_smii.csv',header=None)
    ori_list = ori_df[0].tolist()
    frames = []
    gen_mols = []
    gen_fps = []
    for i in [1024, 2048, 4096, 8192, 16384, 32768]:
        gen_df = pd.read_csv('./sampled_da_info/sampled_da'+str(i)+'_smi.csv', header=None)
        gen_list = gen_df[0].tolist()
        over, num, smi_list = get_smi_list_overlap(ori_list, gen_list)
        smi_mols = get_mols(smi_list)
        smi_fps, failed_mols = get_fingerprints(smi_mols)
        for idx in sorted(failed_mols, reverse=True):
            del smi_list[idx]
        smi_df = pd.Series(data=smi_list, name='SMILES').to_frame()
        smi_df.loc[:,'Group'] = i
        frames.append(smi_df)

    unique_df = pd.concat(frames)
    gen_smi = unique_df['SMILES'].tolist()
    gen_mols = get_mols(gen_smi)
    gen_fps, _ = get_fingerprints(gen_mols)
    unique_df['Gaps'] = predict_property('gbdt_regessor_gap.joblib', gen_fps)
    unique_df['Dips'] = predict_property('gbdt_regessor_dip.joblib', gen_fps)
    promising_df = unique_df.loc[(unique_df['Gaps'] <= 2.0) & (unique_df['Dips']<=2.0)]
    unique_df.to_csv('unique_sampled_smiles_corr2.csv', index=False)
    promising_df.to_csv('Gen_promisings.csv', index=False)


def tsne_projection(train_file, gen_file):
    """
    Creat tsne projection of the fps of the generated promising smiles and the smiles for transfer training.
    Returns: plot of the tsne result

    """
    train_smiles = pd.read_csv(train_file, header=None)
    train_smiles['label'] = 'train'
    gen_prom_smiles = pd.read_csv(gen_file)
    gen_prom_smiles = gen_prom_smiles.drop(['Group', 'Gaps', 'Dips'], axis=1)
    gen_prom_smiles['label'] = 'gen'
    train_smiles.rename(columns={0: "SMILES"}, inplace=True)
    all_smiles = pd.concat([train_smiles, gen_prom_smiles])
    mols = get_mols(all_smiles.SMILES)
    fps, _ = get_fingerprints(mols)
    fp_embeded = TSNE(n_components=2).fit_transform(fps)
    return fp_embeded, len(train_smiles)


def plot_tsne(fps, num_train):
    fig, ax = plot.subplots()
    ax.scatter(fps[:num_train,0], fps[:num_train,1],
               c='blue', s=90,edgecolors='black', linewidths=0.4, label='train')
    ax.scatter(fps[num_train:,0], fps[num_train:,1],
               c='greenyellow', s=90, edgecolors='black', linewidths=0.4, label='generated')
    ax.grid(True)
    ax.axis([-70, 70, -70, 70])
    ax.legend(loc='best',frameon=False, prop={'size':15})
    return ax

def main():
    save_predict_results()
    fp_embeded, num = tsne_projection('refined_smii.csv', 'Gen_promisings.csv')
    ax = plot_tsne(fp_embeded, num_train=num)
    plot.savefig('tsne_plot.png', dpi=300)


if __name__ == '__main__':
    main()



