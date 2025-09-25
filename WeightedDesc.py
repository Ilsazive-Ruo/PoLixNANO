import pandas as pd
import numpy as np


# reading descriptor of SMILES
poly_df = pd.read_csv('data/DemoData.csv')
mono_df = pd.read_csv('data/SmiDesc.csv')
mono_df.drop('SMILES', axis=1, inplace=True)

# set Monomer name as index
mono_df.set_index('Name', inplace=True)


def calc_weighted_descriptors(row):

    mons = []
    degrees = []
    molweights = []

    # reading polymer structures
    for i in [1, 2]:
        mon = row.get(f'm{i}')
        deg = row.get(f'c{i}')
        if pd.isna(mon) or pd.isna(deg):
            continue
        mons.append(mon)
        degrees.append(float(deg))
        molweights.append(float(mono_df.loc[mon, 'MolWt']))

    if not mons:
        return pd.Series([np.nan] * (mono_df.shape[1] - 1))  # 没有单体返回nan

    degrees = np.array(degrees)
    molweights = np.array(molweights)

    total_mass = (degrees * molweights).sum()
    weights = (degrees * molweights) / total_mass

    # calculating weighted descriptors
    desc_cols = [c for c in mono_df.columns if c != 'MolWeight']
    weighted_desc = np.zeros(len(desc_cols))
    for i, mon in enumerate(mons):
        desc_values = mono_df.loc[mon, desc_cols].values.astype(float)
        weighted_desc += weights[i] * desc_values

    return pd.Series(weighted_desc, index=desc_cols)


weighted_by_molweight = poly_df.apply(calc_weighted_descriptors, axis=1)


# weighted_by_molweight.insert(0, 'Polymer', poly_df['Polymer'])
output_df = pd.concat([poly_df, weighted_by_molweight], axis=1)
print(output_df.head())

# output CSV
output_df.to_csv('data/demo-feat.csv', index=False)

print("results have been saved in 'demo-feat.csv'")

# python WeightedDesc.py
