from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


def smiles_to_descriptor_df(smiles_list):
    """
    input:  smiles_list  (list[str])
    output:  pd.DataFrame  — descriptors of each SMILES，name of descriptors were set as columns.
    """
    # access all available descriptors (name, function)
    descriptor_funcs = Descriptors.descList
    names = [name for name, func in descriptor_funcs]

    # saving results
    data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            data.append([None] * len(names))  # invalid SMILESs were fill with None
        else:
            data.append([func(mol) for _, func in descriptor_funcs])

    return pd.DataFrame(data, columns=names)


smis = pd.read_csv('data/structures_prop.csv')
smi_list = smis['SMILES'].values.tolist()
name_list = smis['Name'].values.tolist()

desc_df = smiles_to_descriptor_df(smi_list)
desc_df.drop(columns=desc_df.columns[desc_df.nunique() == 1], inplace=True)
desc_df['Name'] = name_list
desc_df['SMILES'] = smi_list
new_order = ["SMILES", "Name"] + [c for c in desc_df.columns if c not in ["SMILES", "Name"]]
desc_df = desc_df[new_order]
print(desc_df.head())
desc_df.to_csv('data/SmiDesc.csv', index=False)
