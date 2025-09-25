import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             median_absolute_error, mean_absolute_percentage_error)
from xgboost import XGBRegressor


def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    return r2, mae, mse, rmse, mape, medae


def main(csv_path, output_metrics="metrics.csv", output_predictions="predictions.csv",
         model_dir="results/PP", random_state=84):
    df = pd.read_csv(csv_path)
    target_cols = df.columns[9:16]  # particle property columns to prediction
    feature_cols = df.columns[16:]  # feature columns started from 17th column
    X = df[feature_cols].values

    all_metrics = []
    all_predictions = []

    os.makedirs(model_dir, exist_ok=True)

    for i, target in enumerate(target_cols):
        y = df[target].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train).ravel()
        y_test_scaled = y_scaler.transform(y_test).ravel()

        # task specified model selection
        if i == 2 or i == 0:
            model = DecisionTreeRegressor()
            param_grid = {"max_depth": [5, 10, None]}
        elif i == 5 or i == 6:
            model = XGBRegressor(objective="reg:squarederror", random_state=random_state)
            param_grid = {
                "n_estimators": [25, 50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0]
            }
        else:
            model = RandomForestRegressor(random_state=random_state)
            param_grid = {
                "n_estimators": [25, 50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }

        # Pipeline：StandardScale + model prediction
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        # param searching
        if param_grid:
            param_grid_prefixed = {f"model__{k}": v for k, v in param_grid.items()}
            grid = GridSearchCV(pipeline, param_grid_prefixed, cv=5, scoring="r2", n_jobs=-1)
            grid.fit(X_train, y_train_scaled)
            best_model = grid.best_estimator_
        else:
            best_model = pipeline.fit(X_train, y_train_scaled)

        # prediction
        y_train_pred = y_scaler.inverse_transform(best_model.predict(X_train).reshape(-1, 1)).ravel()
        y_test_pred = y_scaler.inverse_transform(best_model.predict(X_test).reshape(-1, 1)).ravel()

        # evaluation
        train_metrics = evaluate_model(y_train.ravel(), y_train_pred)
        test_metrics = evaluate_model(y_test.ravel(), y_test_pred)
        all_metrics.append({
            "Target": target,
            "Train_R2": train_metrics[0],
            "Train_MAE": train_metrics[1],
            "Train_MSE": train_metrics[2],
            "Train_RMSE": train_metrics[3],
            "Train_MAPE": train_metrics[4],
            "Train_MEDAE": train_metrics[5],
            "Test_R2": test_metrics[0],
            "Test_MAE": test_metrics[1],
            "Test_MSE": test_metrics[2],
            "Test_RMSE": test_metrics[3],
            "Test_MAPE": test_metrics[4],
            "Test_MEDAE": test_metrics[5],
            "Best_Params": grid.best_params_
        })

        # save predicted TE index
        df_train_pred = pd.DataFrame({
            "Target": target,
            "Set": "Train",
            "y_true": y_train.ravel(),
            "y_pred": y_train_pred
        })
        df_test_pred = pd.DataFrame({
            "Target": target,
            "Set": "Test",
            "y_true": y_test.ravel(),
            "y_pred": y_test_pred
        })
        all_predictions.append(pd.concat([df_train_pred, df_test_pred], ignore_index=True))

        # ---------------- SHAP & feature importance ----------------
        try:
            raw_model = best_model.named_steps["model"]
            X_scaled = best_model.named_steps["scaler"].transform(X_train)

            if isinstance(raw_model, LinearRegression):
                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Model_Weight": raw_model.coef_
                })
                importance_df["Mean_SHAP"] = np.nan
                importance_df.to_csv(f"{model_dir}/importance-{target}.csv", index=False)

            else:
                explainer = shap.TreeExplainer(raw_model)
                shap_values = explainer.shap_values(X_scaled)
                mean_shap = np.abs(shap_values).mean(axis=0)

                plt.rcParams["font.family"] = "Arial"
                plt.rcParams["font.size"] = 12
                # saving SHAP plots
                plt.figure()
                shap.summary_plot(shap_values, X_scaled, feature_names=feature_cols, max_display=7, show=False,
                                  plot_size=[7, 4])
                plt.tight_layout()
                plt.savefig(f"{model_dir}/shap-{target}.pdf", format="pdf")
                plt.close()

                # save feature importance to CSV
                if hasattr(raw_model, "feature_importances_"):
                    importance_df = pd.DataFrame({
                        "Feature": feature_cols,
                        "Model_Weight": raw_model.feature_importances_,
                        "Mean_SHAP": mean_shap
                    })
                    importance_df.to_csv(f"{model_dir}/importance-{target}.csv", index=False)
        except Exception as e:
            print(f"⚠️ SHAP (Target={target}): {e}")

        # save model weights
        joblib.dump({"model": best_model, "y_scaler": y_scaler}, os.path.join(model_dir, f"{target}.pkl"))

    # save evaluation metrix and predicted values
    pd.DataFrame(all_metrics).to_csv(os.path.join(model_dir, output_metrics), index=False)
    print(pd.DataFrame(all_metrics))
    pd.concat(all_predictions, ignore_index=True).to_csv(os.path.join(model_dir, output_predictions), index=False)

    print(f"✅ results have been saved in {model_dir}/")


if __name__ == "__main__":

    main("data/PP-set.csv")

# python PPPrediction.py
