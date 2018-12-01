import json
import logging
import os
import pprint
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def get_output_dir(args):
    config = json.load(open(args.config))
    output_dir = Path(__file__).parent.parent / config['output_dir'] / Path(args.config).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_logger(args, name, output_dir):
    config = json.load(open(args.config))
    logger = logging.getLogger(Path(name).stem)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(levelname)5s %(asctime)s [%(name)s] %(message)s')
    sc = logging.StreamHandler()
    sc.setFormatter(formatter)
    logger.addHandler(sc)

    fh = logging.FileHandler(output_dir / 'log.txt', mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def show_config(config, logger):
    logger.info('\n' + pprint.pformat(config) + '\n')


def load_dataset(config, filename):
    return pd.read_feather(get_dataset_path(config, filename))


def load_feature(config, logger):
    dfs = []
    for name in config['feature']:
        logger.debug(f'load {name}')
        df = pd.read_feather(os.path.join(config['dataset']['cache_dir'], name + '_train.ftr'))
        logger.debug(df.shape)
        dfs.append(df.set_index('object_id'))
    return pd.concat(dfs, axis=1)


def get_dataset_path(config, filename):
    return os.path.join(config['dataset']['input_dir'], config['dataset']['files'][filename])


def notify(message):
    """Send a notification to LINE chat.
    Please create your access token via https://notify-bot.line.me/my/
    """
    assert type(message) is str
    line_token = "qETBOYfNWwO3hwPDLjApCign4kyVV1s5cjee1B1vlTk"
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = "\n" + message
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(line_notify_api, data=payload, headers=headers)


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def timestamp():
    return time.strftime('%y%m%d_%H%M%S')


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
