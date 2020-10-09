import numpy as np
import pandas as pd

def perform_valuation(df, val_method='dcf', **kwargs):
    '''Performs valuation for all securities in a given DataFrame with earnings data.

    Args:
        df: pd.DataFrame.
            For DCF, df must contain 'date', 'permno', 'beta' and 'ni_i' for i=0,...,n
        val_method: str, default is 'dcf'
        n: int, default is 3. The number of earnings periods to use for dcf. For other valuation, this arg is irrelevant

    Returns:
        pd.DataFrame with two extra columns:
            1. 'vt' of the valuation of the firm at year t (current valuation)
            2. 'v_t1' of the valuation of the firm at year t+1 (next fiscal period valuation)

    Raises:
        ValueError: Placeholder for possible errors
    '''
    if val_method == 'dcf':
        df['PV'] = dcf_valuation(df, **kwargs)
    else:
        raise ValueError(
            f'Invalid valuation method {val_method}.')
    return df


def dcf_valuation(df, n=10, rf=0.02, rp=0.07, tg=0.04):
    """Performs DCF valuation with CAPM discount rate r and terminal growth rate tg.
    :param df: pd.DataFrame
    :param n: int, number of periods
    :param rf: float, risk free rate
    :param rp: float, market premium
    :param tg: float, terminal growth
    :return: np.array of present values
    """
    beta = df['b_mkt']
    r = rf + beta * rp  # CAPM discount rate
    num_firms = df.shape[0]
    PV = np.zeros(num_firms)
    for i in range(n + 1):
        ni_in_period_i = df[f'ni_{i}']
        PV += ni_in_period_i / (1 + r) ** i
        if i == n:
            PV += ni_in_period_i * (1 + tg) / (((1 + r) ** n) * (r - tg + 1e-8))
    return PV
