import sklearn
from sklearn.ensemble import RandomForestRegressor

import pickle 

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import statsmodels as sms
import statsmodels.formula.api as smf

import seaborn as sns # for data visualization
sns.set_style("whitegrid")

from dateutil.relativedelta import *
from pandas.tseries.offsets import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--y', dest='year', action='store',
                    help='Years ahead to forecast NI')

args = parser.parse_args()

yr = args.year

pd.set_option('display.max_columns', None)

mdf2 = pd.read_csv("~/misp_data/lagged_comp_fundr_for_val_1976-2019.csv", index_col=0)

print('mdf2 loaded!!!')

mdf2 = mdf2.replace([np.inf, -np.inf], np.nan)

mdf2 = mdf2.dropna(thresh=int(mdf2.shape[1]/2))

mdf2 = mdf2.fillna(0)

# train val split: 1975-2005 train, 2010-2015 test

dftr = mdf2.loc[(1970 <= mdf2['year']) & (mdf2['year'] <= 2009)]
dfts = mdf2.loc[(2010 <= mdf2['year']) & (mdf2['year'] <= 2015)]

x_vars=['at', 'pstkl', 'txditc', 'pstkrv', 'seq', 'pstk', 'ni', 'epspi', 'revt', 'capx', 'ajex', 'sic2', 'sic', 'naics', 'sale', 'cogs', 'xsga', 'xrd', 'xad', 'ib', 'ebitda', 'ebit', 'nopi', 'spi', 'pi', 'txp', 'txfed', 'txfo', 'txt', 'xint', 'oancf', 'dvt', 'ob', 'gdwlia', 'gdwlip', 'gwo', 'rect', 'act', 'che', 'ppegt', 'invt', 'aco', 'intan', 'ao', 'ppent', 'gdwl', 'fatb', 'fatl', 'lct', 'dlc', 'dltt', 'lt', 'dm', 'dcvt', 'cshrc', 'dcpstk', 'ap', 'lco', 'lo', 'drc', 'drlt', 'txdi', 'ceq', 'scstkc', 'emp', 'csho', 'prcc_f', 'mve_f', 'am', 'txdb', 'dvc', 'dvp', 'dp', 'dvpsx_f', 'mib', 'ivao', 'ivst', 'sstk', 'prstkc', 'dv', 'dltis', 'dltr', 'dlcch', 'oibdp', 'dvpa', 'tstkp', 'oiadp', 'xpp', 'xacc', 're', 'ppenb', 'ppenls', 'capxv', 'fopt', 'wcap', 'be', 'ni_-5', 'at_-5', 'epspi_-5', 'revt_-5', 'capx_-5', 'naics_-5', 'cogs_-5', 'xsga_-5', 'xrd_-5', 'xad_-5', 'ib_-5', 'ebitda_-5', 'ebit_-5', 'nopi_-5', 'pi_-5', 'dvt_-5', 'be_-5', 'ni_-4', 'at_-4', 'epspi_-4', 'revt_-4', 'capx_-4', 'naics_-4', 'cogs_-4', 'xsga_-4', 'xrd_-4', 'xad_-4', 'ib_-4', 'ebitda_-4', 'ebit_-4', 'nopi_-4', 'pi_-4', 'dvt_-4', 'be_-4', 'ni_-3', 'at_-3', 'epspi_-3', 'revt_-3', 'capx_-3', 'naics_-3', 'cogs_-3', 'xsga_-3', 'xrd_-3', 'xad_-3', 'ib_-3', 'ebitda_-3', 'ebit_-3', 'nopi_-3', 'pi_-3', 'dvt_-3', 'be_-3', 'ni_-2', 'at_-2', 'epspi_-2', 'revt_-2', 'capx_-2', 'naics_-2', 'cogs_-2', 'xsga_-2', 'xrd_-2', 'xad_-2', 'ib_-2', 'ebitda_-2', 'ebit_-2', 'nopi_-2', 'pi_-2', 'dvt_-2', 'be_-2', 'ni_-1', 'at_-1', 'epspi_-1', 'revt_-1', 'capx_-1', 'naics_-1', 'cogs_-1', 'xsga_-1', 'xrd_-1', 'xad_-1', 'ib_-1', 'ebitda_-1', 'ebit_-1', 'nopi_-1', 'pi_-1', 'dvt_-1', 'be_-1', 'at_yoy1', 'ni_yoy1', 'epspi_yoy1', 'revt_yoy1', 'capx_yoy1', 'naics_yoy1', 'cogs_yoy1', 'xsga_yoy1', 'xrd_yoy1', 'xad_yoy1', 'ib_yoy1', 'ebitda_yoy1', 'ebit_yoy1', 'nopi_yoy1', 'pi_yoy1', 'dvt_yoy1', 'be_yoy1', 'at_yoy2', 'ni_yoy2', 'epspi_yoy2', 'revt_yoy2', 'capx_yoy2', 'naics_yoy2', 'cogs_yoy2', 'xsga_yoy2', 'xrd_yoy2', 'xad_yoy2', 'ib_yoy2', 'ebitda_yoy2', 'ebit_yoy2', 'nopi_yoy2', 'pi_yoy2', 'dvt_yoy2', 'be_yoy2', 'at_yoy3', 'ni_yoy3', 'epspi_yoy3', 'revt_yoy3', 'capx_yoy3', 'naics_yoy3', 'cogs_yoy3', 'xsga_yoy3', 'xrd_yoy3', 'xad_yoy3', 'ib_yoy3', 'ebitda_yoy3', 'ebit_yoy3', 'nopi_yoy3', 'pi_yoy3', 'dvt_yoy3', 'be_yoy3', 'at_yoy4', 'ni_yoy4', 'epspi_yoy4', 'revt_yoy4', 'capx_yoy4', 'naics_yoy4', 'cogs_yoy4', 'xsga_yoy4', 'xrd_yoy4', 'xad_yoy4', 'ib_yoy4', 'ebitda_yoy4', 'ebit_yoy4', 'nopi_yoy4', 'pi_yoy4', 'dvt_yoy4', 'be_yoy4', 'at_yoy5', 'ni_yoy5', 'epspi_yoy5', 'revt_yoy5', 'capx_yoy5', 'naics_yoy5', 'cogs_yoy5', 'xsga_yoy5', 'xrd_yoy5', 'xad_yoy5', 'ib_yoy5', 'ebitda_yoy5', 'ebit_yoy5', 'nopi_yoy5', 'pi_yoy5', 'dvt_yoy5', 'be_yoy5', 'CAPEI', 'bm', 'evm', 'pe_op_basic', 'pe_op_dil', 'pe_exi', 'pe_inc', 'ps_y', 'pcf', 'dpr', 'npm', 'opmbd', 'opmad', 'gpm', 'ptpm', 'cfm', 'roa', 'roe', 'roce', 'efftax', 'aftret_eq', 'aftret_invcapx', 'aftret_equity', 'pretret_noa', 'pretret_earnat', 'GProf', 'equity_invcap', 'debt_invcap', 'totdebt_invcap', 'capital_ratio', 'int_debt', 'int_totdebt', 'cash_lt', 'invt_act', 'rect_act', 'debt_at', 'debt_ebitda', 'short_debt', 'curr_debt', 'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf', 'lt_ppent', 'dltt_be', 'debt_assets', 'debt_capital', 'de_ratio', 'intcov', 'intcov_ratio', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', 'inv_turn', 'at_turn', 'rect_turn', 'pay_turn', 'sale_invcap', 'sale_equity', 'sale_nwc', 'rd_sale', 'adv_sale', 'staff_sale', 'accrual', 'ptb', 'PEG_trailing', 'PEG_1yrforward', 'PEG_ltgforward']

X_tr = dftr[x_vars].astype(float)
y_tr1 = dftr[f'ni_{yr}'].astype(float)

X_ts = dfts[x_vars].astype(float)
y_ts1 = dfts[f'ni_{yr}'].astype(float)


X_tr1 = X_tr[~(y_tr1==0)]
y_tr1 = y_tr1[~(y_tr1==0)]
print(1, X_tr1.shape, y_tr1.shape)


njobs= -1
max_depth = 10
n_estimators=100
regr1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=njobs, verbose=2)
print(f"{datetime.now()} Fitting regr{yr} start!")
regr1.fit(X_tr1, y_tr1)
print(f"{datetime.now()} Fitting regr{yr} done!")
with open(f'RF_regr{yr}.pickle', 'wb') as handle:
    pickle.dump(regr1, handle, protocol=pickle.HIGHEST_PROTOCOL)

