import os
import sys

import numpy as np
import pandas as pd

if True:
    sys.path.append('..')
    from utils import timer

with timer('train'):
    pd.read_csv('../data/input/train.csv.zip', parse_dates=['first_active_month']).to_feather('../data/input/train.ftr')

with timer('test'):
    pd.read_csv('../data/input/test.csv.zip', parse_dates=['first_active_month']).to_feather('../data/input/test.ftr')

with timer('history'):
    pd.read_csv('../data/input/historical_transactions.csv.zip', parse_dates=['purchase_date']).to_feather('../data/input/history.ftr')

with timer('new'):
    pd.read_csv('../data/input/new_merchant_transactions.csv.zip', parse_dates=['purchase_date']).to_feather('../data/input/new.ftr')

with timer('merchants'):
    pd.read_csv('../data/input/merchants.csv.zip').to_feather('../data/input/merchants.ftr')
