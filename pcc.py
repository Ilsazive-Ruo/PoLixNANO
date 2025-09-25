import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def calc_corr_with_target(df, target_colname, prefix=''):
    target_col = df[target_colname]  # target column
    results = []

    for col in df.columns[8:]:
        if col == target_colname:  # skip target itself
            continue
        # skip Constant column
        if np.std(df[col]) == 0 or np.std(target_col) == 0:
            continue

        # Pearson correlation coefficient
        r_pearson, p_pearson = pearsonr(df[col], target_col)

        # Spearman coefficient
        r_spearman, p_spearman = spearmanr(df[col], target_col)

        results.append({
            "column": col,
            "pearson_r": r_pearson,
            "abs_pearson_r": abs(r_pearson),
            "pearson_p": p_pearson,
            "spearman_r": r_spearman,
            "abs_spearman_r": abs(r_spearman),
            "spearman_p": p_spearman
        })

    res_frame = pd.DataFrame(results)

    # saving results
    res_frame.to_csv(prefix + 'pcc.csv', index=False)

    print('finished target column:', target_colname)
    return res_frame


df = pd.read_csv('data/TE-set.csv')
print(df.head())
targets = ['logTE']
for target in targets:
    calc_corr_with_target(df, target, prefix='TE-')

# python pcc.py
