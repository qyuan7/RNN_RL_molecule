import pandas as pd
import numpy as np
from copy import deepcopy
from rdkit import Chem
from data import *
from sklearn.externals import joblib
from sklearn.manifold import TSNE
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plot


def can_smile(smi_list):
    can_list = []
    for item in smi_list:
        if Chem.MolFromSmiles(item) is not None:
            can_item = Chem.MolToSmiles(Chem.MolFromSmiles(item))
            can_list.append(can_item)
    return can_list

def drop_internal_overlap(fname):

    """
    Drop internal overlap within generated SMILES file
    Args:
        fname: path to the file containing SMILES separated by \n

    Returns: None. Save output file.

    """
    smi_list = []
    with open (fname) as f:
        for line in f:
            smi_list.append(line.rstrip())
    can_smi_list = can_smile(smi_list)
    unique_lst = set(can_smi_list)
    unique_lst = list(unique_lst)
    outf = pd.DataFrame()
    outf['Cano_SMILES'] = pd.Series(data=unique_lst)
    outf.to_csv('Unique_'+fname, index=False)


def get_smi_list_overlap(large, small):
    """

    Args:
        large: list containing the SMILE structures for transfer training
        small: list containing the SMILE structures for transfer sampling

    Returns: num of repeat SMILES, num of unique SMILES in transfer sampling, list of unique SMILES

    """

    large_can, small_can = can_smile(large), can_smile(small)
    small_copy = deepcopy(small_can)
    overlap = set(small_can).intersection(large_can)
    #for item in overlap:
        #list(filter(lambda a: a != item, small_copy))
    small_copy_novel = [item for item in small_copy if item not in large_can]
    return len(overlap), len(small_copy), small_copy_novel


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
    Also save the generated with gap < 2, dip <3.66 as promising candidates
    Returns:

    """

    ori_lst = []
    for i in range(1, 4):
        ori_df = pd.read_csv('Training_Model'+str(i)+'.csv')
        ori_list = ori_df['SMILES'].tolist()
        ori_lst.append(ori_list)
    frames = []
    gen_mols = []
    gen_fps = []
    for i, group in enumerate(['all', 'class3', 'prom']):
        gen_df = pd.read_csv('novel_sampled_cano_script_'+group+'_until.csv')
        gen_list = gen_df['SMILES'].tolist()
        print('Number of molecules in training for model {} is {}'.format(i+1, len(ori_lst[i])))
        over, num, smi_list = get_smi_list_overlap(ori_lst[i], gen_list)
        smi_mols = get_mols(smi_list)
        smi_fps, failed_mols = get_fingerprints(smi_mols)
        for idx in sorted(failed_mols, reverse=True):
            del smi_list[idx]
        smi_df = pd.Series(data=smi_list, name='SMILES').to_frame()
        smi_df.loc[:,'Group'] = i+1
        frames.append(smi_df)

    unique_df = pd.concat(frames)
    gen_smi = unique_df['SMILES'].tolist()
    gen_mols = get_mols(gen_smi)
    gen_fps, _ = get_fingerprints(gen_mols)
    unique_df['Gaps'] = predict_property('gbdt_regessor_gap_regu.joblib', gen_fps)
    unique_df['Dips'] = predict_property('gbdt_regessor_dip_reg.joblib', gen_fps)
    promising_df = unique_df.loc[(unique_df['Gaps'] <= 2.0) & (unique_df['Dips']<=3.66)]
    unique_df.to_csv('Unique_models_15epoch.csv', index=False)
    promising_df.to_csv('Promising_models_15epoch.csv', index=False)


def tsne_projection(train_file, gen_file, i):
    """
    Creat tsne projection of the fps of the generated promising smiles and the smiles for transfer training.
    Returns: plot of the tsne result

    """
    train_data = pd.read_csv(train_file)
    def get_ori_prom(row):
        if row['gaps'] <= 2 and row['dips'] <=2:
            return 'Model'+str(i)+'_train_promising'
        else:
            return 'Model'+str(i)+'_train'
    train_data['label'] = train_data.apply(get_ori_prom, axis=1)
    train_data = train_data.drop(['id', 'gaps', 'dips'], axis=1)
    gen_prom_smiles = pd.read_csv(gen_file)
    gen_prom_smiles = gen_prom_smiles.drop(['Group', 'Gaps', 'Dips'], axis=1)
    gen_prom_smiles['label'] = 'Generated_promising'
    all_smi = pd.concat([train_data, gen_prom_smiles])
    mols = get_mols(all_smi.SMILES)
    fps, _ = get_fingerprints(mols)
    fp_embeded = TSNE(n_components=2, perplexity=100).fit_transform(fps)
    all_smi['tsne1'] = fp_embeded[:, 0]
    all_smi['tsne2'] = fp_embeded[:, 1]
    return all_smi, len(train_data)


def plot_tsne(df, num_train):
    groups = df.groupby('label')
    colors = ['greenyellow', 'blue', 'red']
    fig, ax = plot.subplots()
    ax.set_prop_cycle(color=colors)
    for name, group in sorted(groups):
        ax.scatter(group.tsne1, group.tsne2, marker='o', label=name, edgecolors='black', linewidths=0.4, s=25)
    ax.legend(loc='best', frameon=False, prop={'size': 20})

    ax.grid(True, alpha=0.75)
    #ax.axis([-70, 70, -70, 70])
    ax.legend(loc='best',frameon=False, prop={'size':10})
    return ax

def main():
    save_predict_results()
    #for i in range(1,4):
    #    smi_df, num = tsne_projection('Training_model'+str(i)+'.csv', 'Promising_model'+str(i)+'.csv', i)
    #    ax = plot_tsne(smi_df, num_train=num)
    #    plot.savefig('tsne_plot_model'+str(i)+'.png', dpi=300)
    #for group in ['all', 'class3', 'prom']:
    #    drop_internal_overlap('sampled_da_'+group+'_untill_10000_10epoch.csv')

if __name__ == '__main__':
    main()



