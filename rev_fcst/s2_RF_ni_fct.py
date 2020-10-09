import sklearn
from sklearn.ensemble import RandomForestRegressor

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

pd.set_option('display.max_columns', None)

mdf2 = pd.read_csv("~/misp_data/lagged_comp_fundr_for_val_1976-2019.csv", index_col=0)

print('mdf2 loaded!!!')

mdf2 = mdf2.replace([np.inf, -np.inf], np.nan)

mdf2 = mdf2.dropna(thresh=569)

mdf2 = mdf2.fillna(0)

# train val split: 1975-2005 train, 2005-2010 val, 2010-2015 test

dftr = mdf2.loc[(1970 <= mdf2['year_x']) & (mdf2['year_x'] <= 2015)]
dfts = mdf2.loc[(2016 <= mdf2['year_x']) & (mdf2['year_x'] <= 2019)]

x_vars = ['at', 'pstkl', 'txditc', 'pstkrv', 'seq', 'pstk', 'ni', 'epspi', 'revt', 'capx', 'ajex', 'naics', 'sale', 'cogs', 'xsga', 'xrd', 'xad', 'ib', 'ebitda', 'ebit', 'nopi', 'spi', 'pi', 'txp', 'txfed', 'txfo', 'txt', 'xint', 'oancf', 'dvt', 'ob', 'gdwlia', 'gdwlip', 'gwo', 'rect', 'act', 'che', 'ppegt', 'invt', 'aco', 'intan', 'ao', 'ppent', 'gdwl', 'fatb', 'fatl', 'lct', 'dlc', 'dltt', 'lt', 'dm', 'dcvt', 'cshrc', 'dcpstk', 'ap', 'lco', 'lo', 'drc', 'drlt', 'txdi', 'ceq', 'scstkc', 'emp', 'csho', 'prcc_f', 'mve_f', 'am', 'txdb', 'dvc', 'dvp', 'dp', 'dvpsx_f', 'mib', 'ivao', 'ivst', 'sstk', 'prstkc', 'dv', 'dltis', 'dltr', 'dlcch', 'oibdp', 'dvpa', 'tstkp', 'oiadp', 'xpp', 'xacc', 're', 'ppenb', 'ppenls', 'capxv', 'fopt', 'wcap', 'ps_x', 'be', 'ni_-5', 'at_-5', 'pstkl_-5', 'txditc_-5', 'pstkrv_-5', 'seq_-5', 'pstk_-5', 'epspi_-5', 'revt_-5', 'capx_-5', 'naics_-5', 'sale_-5', 'cogs_-5', 'xsga_-5', 'xrd_-5', 'xad_-5', 'ib_-5', 'ebitda_-5', 'ebit_-5', 'nopi_-5', 'spi_-5', 'pi_-5', 'txp_-5', 'txfed_-5', 'txfo_-5', 'txt_-5', 'xint_-5', 'oancf_-5', 'dvt_-5', 'ob_-5', 'gdwlia_-5', 'gdwlip_-5', 'gwo_-5', 'rect_-5', 'act_-5', 'che_-5', 'ppegt_-5', 'invt_-5', 'aco_-5', 'intan_-5', 'ao_-5', 'ppent_-5', 'gdwl_-5', 'fatb_-5', 'fatl_-5', 'lct_-5', 'dlc_-5', 'dltt_-5', 'lt_-5', 'dm_-5', 'dcvt_-5', 'cshrc_-5', 'dcpstk_-5', 'ap_-5', 'lco_-5', 'lo_-5', 'drc_-5', 'drlt_-5', 'txdi_-5', 'ceq_-5', 'scstkc_-5', 'emp_-5', 'csho_-5', 'prcc_f_-5', 'mve_f_-5', 'am_-5', 'txdb_-5', 'dvc_-5', 'dvp_-5', 'dp_-5', 'dvpsx_f_-5', 'mib_-5', 'ivao_-5', 'ivst_-5', 'sstk_-5', 'prstkc_-5', 'dv_-5', 'dltis_-5', 'dltr_-5', 'dlcch_-5', 'oibdp_-5', 'dvpa_-5', 'tstkp_-5', 'oiadp_-5', 'xpp_-5', 'xacc_-5', 're_-5', 'ppenb_-5', 'ppenls_-5', 'capxv_-5', 'fopt_-5', 'wcap_-5', 'ps_-5', 'be_-5', 'ni_-4', 'at_-4', 'pstkl_-4', 'txditc_-4', 'pstkrv_-4', 'seq_-4', 'pstk_-4', 'epspi_-4', 'revt_-4', 'capx_-4', 'naics_-4', 'sale_-4', 'cogs_-4', 'xsga_-4', 'xrd_-4', 'xad_-4', 'ib_-4', 'ebitda_-4', 'ebit_-4', 'nopi_-4', 'spi_-4', 'pi_-4', 'txp_-4', 'txfed_-4', 'txfo_-4', 'txt_-4', 'xint_-4', 'oancf_-4', 'dvt_-4', 'ob_-4', 'gdwlia_-4', 'gdwlip_-4', 'gwo_-4', 'rect_-4', 'act_-4', 'che_-4', 'ppegt_-4', 'invt_-4', 'aco_-4', 'intan_-4', 'ao_-4', 'ppent_-4', 'gdwl_-4', 'fatb_-4', 'fatl_-4', 'lct_-4', 'dlc_-4', 'dltt_-4', 'lt_-4', 'dm_-4', 'dcvt_-4', 'cshrc_-4', 'dcpstk_-4', 'ap_-4', 'lco_-4', 'lo_-4', 'drc_-4', 'drlt_-4', 'txdi_-4', 'ceq_-4', 'scstkc_-4', 'emp_-4', 'csho_-4', 'prcc_f_-4', 'mve_f_-4', 'am_-4', 'txdb_-4', 'dvc_-4', 'dvp_-4', 'dp_-4', 'dvpsx_f_-4', 'mib_-4', 'ivao_-4', 'ivst_-4', 'sstk_-4', 'prstkc_-4', 'dv_-4', 'dltis_-4', 'dltr_-4', 'dlcch_-4', 'oibdp_-4', 'dvpa_-4', 'tstkp_-4', 'oiadp_-4', 'xpp_-4', 'xacc_-4', 're_-4', 'ppenb_-4', 'ppenls_-4', 'capxv_-4', 'fopt_-4', 'wcap_-4', 'ps_-4', 'be_-4', 'ni_-3', 'at_-3', 'pstkl_-3', 'txditc_-3', 'pstkrv_-3', 'seq_-3', 'pstk_-3', 'epspi_-3', 'revt_-3', 'capx_-3', 'naics_-3', 'sale_-3', 'cogs_-3', 'xsga_-3', 'xrd_-3', 'xad_-3', 'ib_-3', 'ebitda_-3', 'ebit_-3', 'nopi_-3', 'spi_-3', 'pi_-3', 'txp_-3', 'txfed_-3', 'txfo_-3', 'txt_-3', 'xint_-3', 'oancf_-3', 'dvt_-3', 'ob_-3', 'gdwlia_-3', 'gdwlip_-3', 'gwo_-3', 'rect_-3', 'act_-3', 'che_-3', 'ppegt_-3', 'invt_-3', 'aco_-3', 'intan_-3', 'ao_-3', 'ppent_-3', 'gdwl_-3', 'fatb_-3', 'fatl_-3', 'lct_-3', 'dlc_-3', 'dltt_-3', 'lt_-3', 'dm_-3', 'dcvt_-3', 'cshrc_-3', 'dcpstk_-3', 'ap_-3', 'lco_-3', 'lo_-3', 'drc_-3', 'drlt_-3', 'txdi_-3', 'ceq_-3', 'scstkc_-3', 'emp_-3', 'csho_-3', 'prcc_f_-3', 'mve_f_-3', 'am_-3', 'txdb_-3', 'dvc_-3', 'dvp_-3', 'dp_-3', 'dvpsx_f_-3', 'mib_-3', 'ivao_-3', 'ivst_-3', 'sstk_-3', 'prstkc_-3', 'dv_-3', 'dltis_-3', 'dltr_-3', 'dlcch_-3', 'oibdp_-3', 'dvpa_-3', 'tstkp_-3', 'oiadp_-3', 'xpp_-3', 'xacc_-3', 're_-3', 'ppenb_-3', 'ppenls_-3', 'capxv_-3', 'fopt_-3', 'wcap_-3', 'ps_-3', 'be_-3', 'ni_-2', 'at_-2', 'pstkl_-2', 'txditc_-2', 'pstkrv_-2', 'seq_-2', 'pstk_-2', 'epspi_-2', 'revt_-2', 'capx_-2', 'naics_-2', 'sale_-2', 'cogs_-2', 'xsga_-2', 'xrd_-2', 'xad_-2', 'ib_-2', 'ebitda_-2', 'ebit_-2', 'nopi_-2', 'spi_-2', 'pi_-2', 'txp_-2', 'txfed_-2', 'txfo_-2', 'txt_-2', 'xint_-2', 'oancf_-2', 'dvt_-2', 'ob_-2', 'gdwlia_-2', 'gdwlip_-2', 'gwo_-2', 'rect_-2', 'act_-2', 'che_-2', 'ppegt_-2', 'invt_-2', 'aco_-2', 'intan_-2', 'ao_-2', 'ppent_-2', 'gdwl_-2', 'fatb_-2', 'fatl_-2', 'lct_-2', 'dlc_-2', 'dltt_-2', 'lt_-2', 'dm_-2', 'dcvt_-2', 'cshrc_-2', 'dcpstk_-2', 'ap_-2', 'lco_-2', 'lo_-2', 'drc_-2', 'drlt_-2', 'txdi_-2', 'ceq_-2', 'scstkc_-2', 'emp_-2', 'csho_-2', 'prcc_f_-2', 'mve_f_-2', 'am_-2', 'txdb_-2', 'dvc_-2', 'dvp_-2', 'dp_-2', 'dvpsx_f_-2', 'mib_-2', 'ivao_-2', 'ivst_-2', 'sstk_-2', 'prstkc_-2', 'dv_-2', 'dltis_-2', 'dltr_-2', 'dlcch_-2', 'oibdp_-2', 'dvpa_-2', 'tstkp_-2', 'oiadp_-2', 'xpp_-2', 'xacc_-2', 're_-2', 'ppenb_-2', 'ppenls_-2', 'capxv_-2', 'fopt_-2', 'wcap_-2', 'ps_-2', 'be_-2', 'ni_-1', 'at_-1', 'pstkl_-1', 'txditc_-1', 'pstkrv_-1', 'seq_-1', 'pstk_-1', 'epspi_-1', 'revt_-1', 'capx_-1', 'naics_-1', 'sale_-1', 'cogs_-1', 'xsga_-1', 'xrd_-1', 'xad_-1', 'ib_-1', 'ebitda_-1', 'ebit_-1', 'nopi_-1', 'spi_-1', 'pi_-1', 'txp_-1', 'txfed_-1', 'txfo_-1', 'txt_-1', 'xint_-1', 'oancf_-1', 'dvt_-1', 'ob_-1', 'gdwlia_-1', 'gdwlip_-1', 'gwo_-1', 'rect_-1', 'act_-1', 'che_-1', 'ppegt_-1', 'invt_-1', 'aco_-1', 'intan_-1', 'ao_-1', 'ppent_-1', 'gdwl_-1', 'fatb_-1', 'fatl_-1', 'lct_-1', 'dlc_-1', 'dltt_-1', 'lt_-1', 'dm_-1', 'dcvt_-1', 'cshrc_-1', 'dcpstk_-1', 'ap_-1', 'lco_-1', 'lo_-1', 'drc_-1', 'drlt_-1', 'txdi_-1', 'ceq_-1', 'scstkc_-1', 'emp_-1', 'csho_-1', 'prcc_f_-1', 'mve_f_-1', 'am_-1', 'txdb_-1', 'dvc_-1', 'dvp_-1', 'dp_-1', 'dvpsx_f_-1', 'mib_-1', 'ivao_-1', 'ivst_-1', 'sstk_-1', 'prstkc_-1', 'dv_-1', 'dltis_-1', 'dltr_-1', 'dlcch_-1', 'oibdp_-1', 'dvpa_-1', 'tstkp_-1', 'oiadp_-1', 'xpp_-1', 'xacc_-1', 're_-1', 'ppenb_-1', 'ppenls_-1', 'capxv_-1', 'fopt_-1', 'wcap_-1', 'ps_-1', 'be_-1', 'ni_0', 'at_yoy1', 'pstkl_yoy1', 'txditc_yoy1', 'pstkrv_yoy1', 'seq_yoy1', 'pstk_yoy1', 'ni_yoy1', 'epspi_yoy1', 'revt_yoy1', 'capx_yoy1', 'naics_yoy1', 'sale_yoy1', 'cogs_yoy1', 'xsga_yoy1', 'xrd_yoy1', 'xad_yoy1', 'ib_yoy1', 'ebitda_yoy1', 'ebit_yoy1', 'nopi_yoy1', 'spi_yoy1', 'pi_yoy1', 'txp_yoy1', 'txfed_yoy1', 'txfo_yoy1', 'txt_yoy1', 'xint_yoy1', 'oancf_yoy1', 'dvt_yoy1', 'ob_yoy1', 'gdwlia_yoy1', 'gdwlip_yoy1', 'gwo_yoy1', 'rect_yoy1', 'act_yoy1', 'che_yoy1', 'ppegt_yoy1', 'invt_yoy1', 'aco_yoy1', 'intan_yoy1', 'ao_yoy1', 'ppent_yoy1', 'gdwl_yoy1', 'fatb_yoy1', 'fatl_yoy1', 'lct_yoy1', 'dlc_yoy1', 'dltt_yoy1', 'lt_yoy1', 'dm_yoy1', 'dcvt_yoy1', 'cshrc_yoy1', 'dcpstk_yoy1', 'ap_yoy1', 'lco_yoy1', 'lo_yoy1', 'drc_yoy1', 'drlt_yoy1', 'txdi_yoy1', 'ceq_yoy1', 'scstkc_yoy1', 'emp_yoy1', 'csho_yoy1', 'prcc_f_yoy1', 'mve_f_yoy1', 'am_yoy1', 'txdb_yoy1', 'dvc_yoy1', 'dvp_yoy1', 'dp_yoy1', 'dvpsx_f_yoy1', 'mib_yoy1', 'ivao_yoy1', 'ivst_yoy1', 'sstk_yoy1', 'prstkc_yoy1', 'dv_yoy1', 'dltis_yoy1', 'dltr_yoy1', 'dlcch_yoy1', 'oibdp_yoy1', 'dvpa_yoy1', 'tstkp_yoy1', 'oiadp_yoy1', 'xpp_yoy1', 'xacc_yoy1', 're_yoy1', 'ppenb_yoy1', 'ppenls_yoy1', 'capxv_yoy1', 'fopt_yoy1', 'wcap_yoy1', 'ps_yoy1', 'be_yoy1', 'at_yoy2', 'pstkl_yoy2', 'txditc_yoy2', 'pstkrv_yoy2', 'seq_yoy2', 'pstk_yoy2', 'ni_yoy2', 'epspi_yoy2', 'revt_yoy2', 'capx_yoy2', 'naics_yoy2', 'sale_yoy2', 'cogs_yoy2', 'xsga_yoy2', 'xrd_yoy2', 'xad_yoy2', 'ib_yoy2', 'ebitda_yoy2', 'ebit_yoy2', 'nopi_yoy2', 'spi_yoy2', 'pi_yoy2', 'txp_yoy2', 'txfed_yoy2', 'txfo_yoy2', 'txt_yoy2', 'xint_yoy2', 'oancf_yoy2', 'dvt_yoy2', 'ob_yoy2', 'gdwlia_yoy2', 'gdwlip_yoy2', 'gwo_yoy2', 'rect_yoy2', 'act_yoy2', 'che_yoy2', 'ppegt_yoy2', 'invt_yoy2', 'aco_yoy2', 'intan_yoy2', 'ao_yoy2', 'ppent_yoy2', 'gdwl_yoy2', 'fatb_yoy2', 'fatl_yoy2', 'lct_yoy2', 'dlc_yoy2', 'dltt_yoy2', 'lt_yoy2', 'dm_yoy2', 'dcvt_yoy2', 'cshrc_yoy2', 'dcpstk_yoy2', 'ap_yoy2', 'lco_yoy2', 'lo_yoy2', 'drc_yoy2', 'drlt_yoy2', 'txdi_yoy2', 'ceq_yoy2', 'scstkc_yoy2', 'emp_yoy2', 'csho_yoy2', 'prcc_f_yoy2', 'mve_f_yoy2', 'am_yoy2', 'txdb_yoy2', 'dvc_yoy2', 'dvp_yoy2', 'dp_yoy2', 'dvpsx_f_yoy2', 'mib_yoy2', 'ivao_yoy2', 'ivst_yoy2', 'sstk_yoy2', 'prstkc_yoy2', 'dv_yoy2', 'dltis_yoy2', 'dltr_yoy2', 'dlcch_yoy2', 'oibdp_yoy2', 'dvpa_yoy2', 'tstkp_yoy2', 'oiadp_yoy2', 'xpp_yoy2', 'xacc_yoy2', 're_yoy2', 'ppenb_yoy2', 'ppenls_yoy2', 'capxv_yoy2', 'fopt_yoy2', 'wcap_yoy2', 'ps_yoy2', 'be_yoy2', 'at_yoy3', 'pstkl_yoy3', 'txditc_yoy3', 'pstkrv_yoy3', 'seq_yoy3', 'pstk_yoy3', 'ni_yoy3', 'epspi_yoy3', 'revt_yoy3', 'capx_yoy3', 'naics_yoy3', 'sale_yoy3', 'cogs_yoy3', 'xsga_yoy3', 'xrd_yoy3', 'xad_yoy3', 'ib_yoy3', 'ebitda_yoy3', 'ebit_yoy3', 'nopi_yoy3', 'spi_yoy3', 'pi_yoy3', 'txp_yoy3', 'txfed_yoy3', 'txfo_yoy3', 'txt_yoy3', 'xint_yoy3', 'oancf_yoy3', 'dvt_yoy3', 'ob_yoy3', 'gdwlia_yoy3', 'gdwlip_yoy3', 'gwo_yoy3', 'rect_yoy3', 'act_yoy3', 'che_yoy3', 'ppegt_yoy3', 'invt_yoy3', 'aco_yoy3', 'intan_yoy3', 'ao_yoy3', 'ppent_yoy3', 'gdwl_yoy3', 'fatb_yoy3', 'fatl_yoy3', 'lct_yoy3', 'dlc_yoy3', 'dltt_yoy3', 'lt_yoy3', 'dm_yoy3', 'dcvt_yoy3', 'cshrc_yoy3', 'dcpstk_yoy3', 'ap_yoy3', 'lco_yoy3', 'lo_yoy3', 'drc_yoy3', 'drlt_yoy3', 'txdi_yoy3', 'ceq_yoy3', 'scstkc_yoy3', 'emp_yoy3', 'csho_yoy3', 'prcc_f_yoy3', 'mve_f_yoy3', 'am_yoy3', 'txdb_yoy3', 'dvc_yoy3', 'dvp_yoy3', 'dp_yoy3', 'dvpsx_f_yoy3', 'mib_yoy3', 'ivao_yoy3', 'ivst_yoy3', 'sstk_yoy3', 'prstkc_yoy3', 'dv_yoy3', 'dltis_yoy3', 'dltr_yoy3', 'dlcch_yoy3', 'oibdp_yoy3', 'dvpa_yoy3', 'tstkp_yoy3', 'oiadp_yoy3', 'xpp_yoy3', 'xacc_yoy3', 're_yoy3', 'ppenb_yoy3', 'ppenls_yoy3', 'capxv_yoy3', 'fopt_yoy3', 'wcap_yoy3', 'ps_yoy3', 'be_yoy3', 'at_yoy4', 'pstkl_yoy4', 'txditc_yoy4', 'pstkrv_yoy4', 'seq_yoy4', 'pstk_yoy4', 'ni_yoy4', 'epspi_yoy4', 'revt_yoy4', 'capx_yoy4', 'naics_yoy4', 'sale_yoy4', 'cogs_yoy4', 'xsga_yoy4', 'xrd_yoy4', 'xad_yoy4', 'ib_yoy4', 'ebitda_yoy4', 'ebit_yoy4', 'nopi_yoy4', 'spi_yoy4', 'pi_yoy4', 'txp_yoy4', 'txfed_yoy4', 'txfo_yoy4', 'txt_yoy4', 'xint_yoy4', 'oancf_yoy4', 'dvt_yoy4', 'ob_yoy4', 'gdwlia_yoy4', 'gdwlip_yoy4', 'gwo_yoy4', 'rect_yoy4', 'act_yoy4', 'che_yoy4', 'ppegt_yoy4', 'invt_yoy4', 'aco_yoy4', 'intan_yoy4', 'ao_yoy4', 'ppent_yoy4', 'gdwl_yoy4', 'fatb_yoy4', 'fatl_yoy4', 'lct_yoy4', 'dlc_yoy4', 'dltt_yoy4', 'lt_yoy4', 'dm_yoy4', 'dcvt_yoy4', 'cshrc_yoy4', 'dcpstk_yoy4', 'ap_yoy4', 'lco_yoy4', 'lo_yoy4', 'drc_yoy4', 'drlt_yoy4', 'txdi_yoy4', 'ceq_yoy4', 'scstkc_yoy4', 'emp_yoy4', 'csho_yoy4', 'prcc_f_yoy4', 'mve_f_yoy4', 'am_yoy4', 'txdb_yoy4', 'dvc_yoy4', 'dvp_yoy4', 'dp_yoy4', 'dvpsx_f_yoy4', 'mib_yoy4', 'ivao_yoy4', 'ivst_yoy4', 'sstk_yoy4', 'prstkc_yoy4', 'dv_yoy4', 'dltis_yoy4', 'dltr_yoy4', 'dlcch_yoy4', 'oibdp_yoy4', 'dvpa_yoy4', 'tstkp_yoy4', 'oiadp_yoy4', 'xpp_yoy4', 'xacc_yoy4', 're_yoy4', 'ppenb_yoy4', 'ppenls_yoy4', 'capxv_yoy4', 'fopt_yoy4', 'wcap_yoy4', 'ps_yoy4', 'be_yoy4', 'at_yoy5', 'pstkl_yoy5', 'txditc_yoy5', 'pstkrv_yoy5', 'seq_yoy5', 'pstk_yoy5', 'ni_yoy5', 'epspi_yoy5', 'revt_yoy5', 'capx_yoy5', 'naics_yoy5', 'sale_yoy5', 'cogs_yoy5', 'xsga_yoy5', 'xrd_yoy5', 'xad_yoy5', 'ib_yoy5', 'ebitda_yoy5', 'ebit_yoy5', 'nopi_yoy5', 'spi_yoy5', 'pi_yoy5', 'txp_yoy5', 'txfed_yoy5', 'txfo_yoy5', 'txt_yoy5', 'xint_yoy5', 'oancf_yoy5', 'dvt_yoy5', 'ob_yoy5', 'gdwlia_yoy5', 'gdwlip_yoy5', 'gwo_yoy5', 'rect_yoy5', 'act_yoy5', 'che_yoy5', 'ppegt_yoy5', 'invt_yoy5', 'aco_yoy5', 'intan_yoy5', 'ao_yoy5', 'ppent_yoy5', 'gdwl_yoy5', 'fatb_yoy5', 'fatl_yoy5', 'lct_yoy5', 'dlc_yoy5', 'dltt_yoy5', 'lt_yoy5', 'dm_yoy5', 'dcvt_yoy5', 'cshrc_yoy5', 'dcpstk_yoy5', 'ap_yoy5', 'lco_yoy5', 'lo_yoy5', 'drc_yoy5', 'drlt_yoy5', 'txdi_yoy5', 'ceq_yoy5', 'scstkc_yoy5', 'emp_yoy5', 'csho_yoy5', 'prcc_f_yoy5', 'mve_f_yoy5', 'am_yoy5', 'txdb_yoy5', 'dvc_yoy5', 'dvp_yoy5', 'dp_yoy5', 'dvpsx_f_yoy5', 'mib_yoy5', 'ivao_yoy5', 'ivst_yoy5', 'sstk_yoy5', 'prstkc_yoy5', 'dv_yoy5', 'dltis_yoy5', 'dltr_yoy5', 'dlcch_yoy5', 'oibdp_yoy5', 'dvpa_yoy5', 'tstkp_yoy5', 'oiadp_yoy5', 'xpp_yoy5', 'xacc_yoy5', 're_yoy5', 'ppenb_yoy5', 'ppenls_yoy5', 'capxv_yoy5', 'fopt_yoy5', 'wcap_yoy5', 'ps_yoy5', 'be_yoy5', 'CAPEI', 'bm', 'evm', 'pe_op_basic', 'pe_op_dil', 'pe_exi', 'pe_inc', 'ps_y', 'pcf', 'dpr', 'npm', 'opmbd', 'opmad', 'gpm', 'ptpm', 'cfm', 'roa', 'roe', 'roce', 'efftax', 'aftret_eq', 'aftret_invcapx', 'aftret_equity', 'pretret_noa', 'pretret_earnat', 'GProf', 'equity_invcap', 'debt_invcap', 'totdebt_invcap', 'capital_ratio', 'int_debt', 'int_totdebt', 'cash_lt', 'invt_act', 'rect_act', 'debt_at', 'debt_ebitda', 'short_debt', 'curr_debt', 'lt_debt', 'profit_lct', 'ocf_lct', 'cash_debt', 'fcf_ocf', 'lt_ppent', 'dltt_be', 'debt_assets', 'debt_capital', 'de_ratio', 'intcov', 'intcov_ratio', 'cash_ratio', 'quick_ratio', 'curr_ratio', 'cash_conversion', 'inv_turn', 'at_turn', 'rect_turn', 'pay_turn', 'sale_invcap', 'sale_equity', 'sale_nwc', 'rd_sale', 'adv_sale', 'staff_sale', 'accrual', 'ptb', 'PEG_trailing', 'PEG_1yrforward', 'PEG_ltgforward']

X_tr = dftr[x_vars].astype(float)
y_tr1 = dftr['ni_1'].astype(float)
y_tr2 = dftr['ni_2'].astype(float)
y_tr3 = dftr['ni_3'].astype(float)
y_tr4 = dftr['ni_4'].astype(float)
y_tr5 = dftr['ni_5'].astype(float)

X_ts = dfts[x_vars].astype(float)
y_ts1 = dfts['ni_1'].astype(float)
y_ts2 = dfts['ni_2'].astype(float)
y_ts3 = dfts['ni_3'].astype(float)
y_ts4 = dfts['ni_4'].astype(float)
y_ts5 = dfts['ni_5'].astype(float)


X_tr1 = X_tr[~(y_tr1==0)]
y_tr1 = y_tr1[~(y_tr1==0)]
print(1, X_tr1.shape, y_tr1.shape)

X_tr2 = X_tr[~(y_tr2==0)]
y_tr2 = y_tr2[~(y_tr2==0)]
print(2, X_tr2.shape, y_tr2.shape)

X_tr3 = X_tr[~(y_tr3==0)]
y_tr3 = y_tr3[~(y_tr3==0)]
print(3, X_tr3.shape, y_tr3.shape)

X_tr4 = X_tr[~(y_tr4==0)]
y_tr4 = y_tr4[~(y_tr4==0)]
print(4, X_tr4.shape, y_tr4.shape)

X_tr5 = X_tr[~(y_tr5==0)]
y_tr5 = y_tr5[~(y_tr5==0)]
print(5, X_tr5.shape, y_tr5.shape)

njobs= 128
max_depth = 10
n_estimators=200
regr1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=njobs, verbose=2)
print(f"{datetime.now()} Fitting regr1 start!")
regr1.fit(X_tr1, y_tr1)
print(f"{datetime.now()} Fitting regr1 done!")
with open('RF_regr1.pickle', 'wb') as handle:
    pickle.dump(regr1, handle, protocol=pickle.HIGHEST_PROTOCOL)


regr2 = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, n_jobs=njobs, verbose=1)
regr2.fit(X_tr2, y_tr2)
print(f"{datetime.now()} Fitting regr2 done!")
with open('RF_regr2.pickle', 'wb') as handle:
    pickle.dump(regr2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
regr3 = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, n_jobs=njobs, verbose=1)
regr3.fit(X_tr3, y_tr3)
print(f"{datetime.now()} Fitting regr3 done!")
with open('RF_regr3.pickle', 'wb') as handle:
    pickle.dump(regr3, handle, protocol=pickle.HIGHEST_PROTOCOL)


regr4 = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, n_jobs=njobs, verbose=1)
regr4.fit(X_tr4, y_tr4)
print(f"{datetime.now()} Fitting regr4 done!")
with open('RF_regr4.pickle', 'wb') as handle:
    pickle.dump(regr4, handle, protocol=pickle.HIGHEST_PROTOCOL)

regr5 = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, n_jobs=njobs, verbose=1)
regr5.fit(X_tr5, y_tr5)
print(f"{datetime.now()} Fitting regr5 done!")
with open('RF_regr5.pickle', 'wb') as handle:
    pickle.dump(regr5, handle, protocol=pickle.HIGHEST_PROTOCOL)


