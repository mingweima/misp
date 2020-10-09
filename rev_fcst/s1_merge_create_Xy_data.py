import sklearn
from sklearn.ensemble import RandomForestRegressor

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


fundr = pd.read_csv('~/misp_data/fund_ratios_1970-2020.csv', index_col=0)
comp = pd.read_csv('~/misp_data/comp_1959-2019.csv', index_col=0)

# import fund_ratio data since 1970

from dateutil.relativedelta import *
from pandas.tseries.offsets import *

fund_r = fundr[fundr['qdate'].notna()]
fund_r['date'] = pd.to_datetime((fund_r['qdate']))
fund_r['jdate']=fund_r['date']+MonthEnd(0)
fund_r['year'] = fund_r['jdate'].dt.year


compvars = [ 'at',
                     'pstkl',
                     'txditc',
                     'pstkrv',
                     'seq',
                     'pstk',
                     'ni',
                     'epspi',
                     'revt',
                     'capx',
                     'naics',
                     'sale',
                     'cogs',
                     'xsga',
                     'xrd',
                     'xad',
                     'ib',
                     'ebitda',
                     'ebit',
                     'nopi',
                     'spi',
                     'pi',
                     'txp',
                     'txfed',
                     'txfo',
                     'txt',
                     'xint',
                     'oancf',
                     'dvt',
                     'ob',
                     'gdwlia',
                     'gdwlip',
                     'gwo',
                     'rect',
                     'act',
                     'che',
                     'ppegt',
                     'invt',
                     'aco',
                     'intan',
                     'ao',
                     'ppent',
                     'gdwl',
                     'fatb',
                     'fatl',
                     'lct',
                     'dlc',
                     'dltt',
                     'lt',
                     'dm',
                     'dcvt',
                     'cshrc',
                     'dcpstk',
                     'ap',
                     'lco',
                     'lo',
                     'drc',
                     'drlt',
                     'txdi',
                     'ceq',
                     'scstkc',
                     'emp',
                     'csho',
                     'prcc_f',
                     'mve_f',
                     'am',
                     'txdb',
                     'dvc',
                     'dvp',
                     'dp',
                     'dvpsx_f',
                     'mib',
                     'ivao',
                     'ivst',
                     'sstk',
                     'prstkc',
                     'dv',
                     'dltis',
                     'dltr',
                     'dlcch',
                     'oibdp',
                     'dvpa',
                     'tstkp',
                     'oiadp',
                     'xpp',
                     'xacc',
                     're',
                     'ppenb',
                     'ppenls',
                     'capxv',
                     'fopt',
                     'wcap',
                     'ps',
                     'be',]


mdf = comp
mdf = mdf.sort_values(['permno', 'year'], ascending=[True, False])

eng_yr = 10
pred_var_yr = 5
pred_i = list(range(-pred_var_yr, 0))

for i in range(-pred_var_yr, eng_yr+1):
    mdf[f'ni_{i}'] = mdf.groupby('permno')['ni'].shift(i)
    print(i)
    if i in pred_i:
        for var in compvars:
            mdf[f'{var}_{i}'] = mdf.groupby('permno')[var].shift(i)
            
for i in range(1, 6):
    print(i)
    for var in compvars: 
        mdf[f'{var}_yoy{i}'] = ((mdf[f'{var}'] - mdf[f'{var}_-{i}']) / mdf[f'{var}'])**(1/i)
        
        
mdf['jdate'] = pd.to_datetime((mdf['jdate']), format='%Y-%m-%d')
mdf = mdf.merge(fund_r, how='left', on=['permno', 'jdate'])

mdf.to_csv('~/misp_data/lagged_comp_fundr_for_val_1976-2019.csv')
