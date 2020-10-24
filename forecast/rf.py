from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import statsmodels as sms
import statsmodels.formula.api as smf
from scipy import stats

n_estimators = 200
max_depth = 4

DATA_FOLDER = '/Users/mmw/data/misp_data/'
data = pd.read_csv(f'{DATA_FOLDER}/lagged_comp-fundr-ibes_for_val_1976-2019.csv', index_col = 0)


data = data.replace([np.inf, -np.inf], np.nan)
# train val split: 1979-2009 train, 2010-2019 test
tr = data.loc[(1979 <= data['year']) & (data['year'] <= 2009)]
ts = data.loc[(2010 <= data['year']) & (data['year'] <= 2019)]

for yr in [1,2,3,4,5]:
    tryr = tr.dropna(subset = [f'ni_{yr}'])
    X_tr = tryr.drop(['ticker', 'exchcd', 'permno','jdate','year','ni_1', 'ni_2', 'ni_3','ni_4','ni_5'], axis = 1)
    y_tr = tryr[f'ni_{yr}']

    X_tr = X_tr.fillna(0).astype(float)

    y_tr = y_tr.astype(float)

    print(yr, X_tr.shape, y_tr.shape)
    
    rf1 = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, verbose=2)    
    print(f"{datetime.now()} Fitting RF regr{yr} start!")
    rf1.fit(X_tr, y_tr)
    print(f"{datetime.now()} Fitting RF regr{yr} done!")
    with open(f'RF_regr{yr}_tree{n_estimators}_depth{max_depth}.pickle', 'wb') as handle:
        pickle.dump(rf1, handle, protocol=pickle.HIGHEST_PROTOCOL)
   