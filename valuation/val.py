import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from global_import import *

def perform_valuation(df, val_method='dcf', **kwargs):
    if val_method == 'dcf':
        df['PV'] = dcf_valuation(df, **kwargs)
    else:
        raise ValueError(
            f'Invalid valuation method {val_method}.')
    return df


def dcf_valuation(df, n=10, rp=0.06, tg=0.06):
    beta = df['b_mkt']
    rf = df['RF']
    r = rf + beta * rp  # CAPM discount rate
    num_firms = df.shape[0]
    PV = np.zeros(num_firms)
    for i in range(n):
        ni_in_period_i = df[f'ni_{i+1}']
        PV += ni_in_period_i / (1 + r) ** i
        if i == n-1:
            PV += ni_in_period_i * (1 + tg) / (((1 + r) ** n) * (r - tg + 1e-10))
    return PV