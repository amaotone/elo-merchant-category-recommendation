import os
import sys

import numpy as np
import pandas as pd

if True:
    sys.path.append('..')
    from utils import timer, reduce_mem_usage

with timer('train'):
    reduce_mem_usage(pd.read_csv('../data/input/train.csv.zip', parse_dates=['first_active_month'])
                     .replace({'Y': True, 'N': False, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}))\
        .reset_index(drop=True).to_feather('../data/input/train.ftr')

with timer('test'):
    reduce_mem_usage(pd.read_csv('../data/input/test.csv.zip', parse_dates=['first_active_month'])
                     .replace({'Y': True, 'N': False, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}))\
        .reset_index(drop=True).to_feather('../data/input/test.ftr')

with timer('old'):
    reduce_mem_usage(pd.read_csv('../data/input/historical_transactions.csv.zip', parse_dates=['purchase_date'])
                     .replace({'Y': True, 'N': False, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}))\
        .reset_index(drop=True).to_feather('../data/input/old.ftr')

with timer('new'):
    reduce_mem_usage(pd.read_csv('../data/input/new_merchant_transactions.csv.zip', parse_dates=['purchase_date'])
                     .replace({'Y': True, 'N': False, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}))\
        .reset_index(drop=True).to_feather('../data/input/new.ftr')

with timer('merchants'):
    reduce_mem_usage(pd.read_csv('../data/input/merchants.csv.zip')
                     .replace({'Y': True, 'N': False, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}))\
        .reset_index(drop=True).to_feather('../data/input/merchants.ftr')
