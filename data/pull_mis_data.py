from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
from datetime import datetime

from cfgs.wrds_conn import conn

def pull_wrds_data(datadate, df, conn=conn):
    '''Pull relevant WRDS data for all CRSP securities in a given date.
        For COMPUSTATS, this includes:

    Args:
        datadate: str, 'yyyy-mm-dd', a specific date for which to pull backward-looking data from WRDS
        df: pd.DataFrame, contains 'permno'
        conn: wrds.Connection

    Returns:
        pd.DataFrame with extra columns containing data pulled for the date

    Raises:
        PlaceHolderError: Placeholder for possible errors
    '''
    datadate_stamp = pd.to_datetime(datadate)
    year = datadate_stamp.dt.year

    comp = conn.raw_sql(f"""
                        select gvkey, datadate, at, pstkl, txditc,
                        pstkrv, seq, pstk
                        from comp.funda
                        where indfmt='INDL' 
                        and datafmt='STD'
                        and popsrc='D'
                        and consol='C'
                        and datadate = {datadate}
                        """)