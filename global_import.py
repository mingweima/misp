import numpy as np
import pandas as pd
import statsmodels as sms
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import seaborn as sns # for data visualization
sns.set_style("whitegrid")

from dateutil.relativedelta import *
from pandas.tseries.offsets import *

pd.set_option('display.max_columns', None)
DATA_FOLDER = '/Users/mmw/data/misp_data'
PROJECT_FOLDER = '/Users/mmw/Documents/GitHub/misp'

