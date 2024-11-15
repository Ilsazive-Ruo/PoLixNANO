import random
import joblib
import numpy as np
import pandas as pd


def simple2x(simple):
    simple_x = simple[1:5] + structure_props[simple[5]] + [simple[6]] + structure_props[simple[7]] + [simple[8]]
    return np.array(simple_x)


structure_props = {}
SP_data = pd.read_csv('data/structures_prop.csv')
for Num in SP_data['Num'].values:
    structure_props[Num] = SP_data.iloc[Num, 4:].values.tolist()


data = pd.read_csv('data/prediction_source.csv')
x = []
for i in range(len(data)):
    x.append(simple2x(data.iloc[i, :].tolist()))

x = np.array(x)

model_size = joblib.load('data/Size_XGB.pkl')
model_PDI = joblib.load('data/PDI_XGB.pkl')
model_NDI = joblib.load('data/NDI_MultiRF.pkl')
model_DC = joblib.load('data/DC_XGB.pkl')
model_EE = joblib.load('data/EE_MultiXGB.pkl')
sclaer = joblib.load('data/scaler.joblib')

Size_pred = model_size.predict(x)
PDI_pred = model_PDI.predict(x)
NDI_pred = model_NDI.predict(x)[:, 0]
DC_pred = model_DC.predict(x)
EE_pred = model_EE.predict(x)[:, 3]

x2 = []
for i in range(Size_pred.shape[0]):
    x2.append([NDI_pred[i], Size_pred[i], PDI_pred[i], EE_pred[i], DC_pred[i]])
x2 = np.array(x2)
feature = sclaer.transform(x2)

TE = []
for y_pred in feature:
    TE.append(5.45 + (-0.14 * y_pred[0]) + (-0.01 * y_pred[1]) + (-0.05 * y_pred[2]) + (0.23 * y_pred[3])
              + (0.25 * y_pred[4]))


res = {}
for i in range(len(data)):
    res[str(i)] = data.iloc[i, :].tolist() + list(x2[i, :]) + [TE[i]]

res = pd.DataFrame(res).transpose()
res.columns = data.columns.tolist() + ['NDI', 'Size', 'PDI', 'EE', 'DC', 'TE']
print(res)
res.to_csv('data/prediction.csv', index=False)
