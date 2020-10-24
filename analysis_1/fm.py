import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from global_import import *


def ols_coef(x,formula):
    return smf.ols(formula,data=x).fit().params


def fama_macbeth_summary(p):
    s = p.describe().T
    s['std_error'] = s['std']/np.sqrt(s['count'])
    s['tstat'] = s['mean']/s['std_error']
    return s[['count', 'mean','std_error','tstat',]]

def ts_nw(xt):
    e = xt - np.mean(xt)
    T = e.shape[0]
    L = int(4*(T/100)**(2/9))
    w = 0
    for l in range(1, L+1):
        wl = 1-(l/(1+L))
        for t in range(l+1, T+1):
            w+= 2* wl * e[t-1] * e[t-l-1]
    S = w + np.sum(e**2)
    return np.sqrt(S)/T

def fama_macbeth_reg_panel_nw(regdf, xname='misp', yname='ret', 
                               csname='permno', tsname='jdate'): 
    # Panel factor setting FM reg
    gamma_cs = (regdf.groupby(tsname).apply(ols_coef,f'{yname} ~ {xname}'))
    gamma_cs = gamma_cs.rename(columns={"Intercept": "alpha_i_t", f"beta_{xname}": "lamba_t"})
    summary = fama_macbeth_summary(gamma_cs)
    nw_std_error_alpha = ts_nw(np.array(gamma_cs['alpha_i_t']))
    nw_std_error_beta = ts_nw(np.array(gamma_cs[xname]))
    summary['std_error_nw'] = np.array([nw_std_error_alpha, nw_std_error_beta])
    summary['tstat_nw'] = summary['mean'] / np.array([nw_std_error_alpha, nw_std_error_beta])
    return summary