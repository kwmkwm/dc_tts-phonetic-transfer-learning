#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function
from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
import tqdm
from hyperparams import Hyperparams as hp
import argparse
import sys


def get_arguments():
    parser = argparse.ArgumentParser(description='DC_TTS pre-processor')
    parser.add_argument('--data', type=str, required=False,
                        help='Directory containing training data')
    arguments = parser.parse_args()
    return arguments


def prepro(datadir):
    # Load data
    fpaths, _, _ = load_data(datadir)  # list

    for fpath in tqdm.tqdm(fpaths):
        fname, mel, mag = load_spectrograms(fpath)
        if not os.path.exists(datadir + hp.meldir):
            os.makedirs(datadir + hp.meldir)
        if not os.path.exists(datadir + hp.magdir):
            os.makedirs(datadir + hp.magdir)

        np.save(datadir + hp.meldir +
                "/{}".format(fname.replace("wav", "npy")), mel)
        np.save(datadir + hp.magdir +
                "/{}".format(fname.replace("wav", "npy")), mag)


if __name__ == '__main__':
    args = get_arguments()

    if args.data:
        if not os.path.exists(args.data):
            print('Directory %s not found. Exiting.' % args.data)
            sys.exit()
        else:
            datadir = args.data
    else:
        datadir = hp.data

    prepro(datadir)
