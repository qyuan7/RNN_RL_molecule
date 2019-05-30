#!/usr/bin/env python
#noinspection PyUnresolvedReferences
import sqlite3
from rdkit import Chem
from rdkit.Chem import AllChem


def connect_db(db_file, parameter):
    """
    Execute SQL query to obtain molecules and parameters
    Args:
        db_file: database for examination
        parameter: molecule parameter for query

    Returns: query result containing SMILES for molecules and their properties

    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    t = (parameter,)
    c.execute("""SELECT text_key_values.key,text_key_values.value,text_key_values.id,
            number_key_values.key, number_key_values.value
            FROM text_key_values INNER JOIN number_key_values
            ON text_key_values.id=number_key_values.id 
            WHERE text_key_values.key='SMILES' and number_key_values.key=?""", t)
    result = c.fetchall()
    return result


def get_data(query_result):
    """
    Retrive lists of SMILES, compound unique ids and parameters from SQL query result
    Args:
        query_result: result for SQL query containing molecular information and property

    Returns: lists of smiles, compound uIDs and parameters

    """
    smiles = [item[1].rstrip() for item in query_result]
    compounds = [item[2] for item in query_result]
    gaps = [item[-1] for item in query_result]
    return smiles, compounds, gaps


def get_mols(smiles):
    """
    Compute the mols from SMILES files
    Args:
        smiles: list of SMILES files

    Returns: list of mols for the candidate molecules

    """
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    return mols


def get_fingerprints(mols):
    """
    get morgan fingerprints for the molecules corresponding to the mols and index for unreasonable mols
    Args:
        mols: list of mols for the molecules

    Returns: list of molecule fingerprints, list of indexes for failed mols

    """
    fps_morgan = []; failed_mols = []
    for i in range(len(mols)):
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mols[i],2,1024) # radius and bit_size can be modified
            fps_morgan.append(fp)
        except:
            failed_mols.append(i)
    return fps_morgan, failed_mols


def refine_compounds(compounds, mols, gaps, failed_mols):
    """
    Remove molecules with which reasonable mols cannot be generated
    Args:
        compounds: list of compound unique IDs
        mols: list of mols for compounds
        gaps: list of energy gaps for compounds
        failed_mols: list of indexes for failed compounds

    Returns: None

    """
    for idx in sorted(failed_mols, reverse=True):
        del compounds[idx]
        del mols[idx]
        del gaps[idx]

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

