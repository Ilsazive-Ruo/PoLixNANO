import pandas as pd
import numpy as np
import joblib
import argparse
import os


def cascade_predict_task1_7(
        unknown_csv,
        PP_weights_dir,
        TE_weight_path,
        output_dir="results"
):
    """
    Predictor for end-to-end prediction, from molecular descriptors to TE index：
    """
    df = pd.read_csv(unknown_csv)

    if not os.path.exists(output_dir):
        print("dir not exists，building...")
        os.makedirs(output_dir)
    else:
        print("dir existed")
    # input features for PP prediction
    task1_feature_cols = df.columns[6:]
    X_task1 = df[task1_feature_cols].values

    weight_list = ['Size.pkl', 'PDI.pkl', 'EE.pkl', 'NDI-Size.pkl', 'NDI-PDI.pkl', 'NDI-EE.pkl', 'Deff.pkl']
    # loading PP prediction weights
    T1_weights = {}
    T1_scalers = {}
    for file_name in weight_list:
        file_path = os.path.join(PP_weights_dir, file_name)
        loaded_dict = joblib.load(file_path)
        T1_weights[file_name] = loaded_dict['model']
        T1_scalers[file_name] = loaded_dict['y_scaler']

    # PP prediction targets(stage 2)
    target_cols_task1 = ['Size', 'PDI', 'EE', 'NDI-Size', 'NDI-PDI', 'NDI-EE', 'Deff']

    # PP prediction
    task1_preds = []
    for target in weight_list:
        y_scaler = T1_scalers[target]
        pred_scaled = T1_weights[target].predict(X_task1)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        task1_preds.append(pred)
    task1_preds_array = np.column_stack(task1_preds)  # n_samples x 7

    # TE prediction input = PP prediction output + TE prediction input
    X_task2 = np.concatenate([task1_preds_array, X_task1], axis=1)

    # Loading TE prediction models (stage 1)
    task2_data = joblib.load(TE_weight_path)
    task2_model = task2_data["model"]
    task2_y_scaler = task2_data["y_scaler"]

    # TE prediction
    task2_pred_scaled = task2_model.predict(X_task2)
    task2_pred = task2_y_scaler.inverse_transform(task2_pred_scaled.reshape(-1, 1)).ravel()

    # output DataFrame
    output_df = df[['Num.', 'polymer name', 'm1', 'c1', 'm2', 'c2']].copy()
    for i, target in enumerate(target_cols_task1):
        output_df[f"{target}_pred"] = task1_preds_array[:, i]
    output_df['logTE_pred'] = task2_pred

    # saving results
    output_df.to_csv(output_dir + '/res.csv', index=False)
    print(f"✅ results have been saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="end-to-end predictor")
    parser.add_argument("-i", "--input", required=True, help="custom CSV file path")
    parser.add_argument("-PP", "--stage2_model", required=True,
                        help="PP prediction model weights (7 models + scalers)")
    parser.add_argument("-TE", "--stage1_model", required=True,
                        help="TE prediction model weights")
    parser.add_argument("-o", "--output", default="res", help="prediction output path")

    args = parser.parse_args()

    cascade_predict_task1_7(
        unknown_csv=args.input,
        PP_weights_dir=args.stage2_model,
        TE_weight_path=args.stage1_model,
        output_dir=args.output
    )

# python Predictor.py -i data/demo-feat.csv -PP results/PP -TE results/TE/modelTE.pkl -o results/demo
