#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hyperparams import Hyperparams as hp
from data_load import text_normalize
from data_load import phoneme_normalize
from data_load import load_phns
from g2p_en import G2p
import argparse
import numpy as np

g2p = G2p()

def get_arguments():
    parser = argparse.ArgumentParser(description='test normalization of text and phonemes')
    parser.add_argument('text', type=str, help='text string to normalize')
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_arguments()

    print("Entered Text:")
    print(args.text)
    ntext = text_normalize(args.text)
    print("Normalized Text:")
    print(ntext)
    phones = g2p(ntext)
    print("Phonemes:")
    print(phones)
    nphones = phoneme_normalize(phones,True)
    print("Normalized Phonemes:")
    print(nphones)
    phn2idx, idx2phn = load_phns()
    indexed = [phn2idx[phn] for phn in nphones]
    print("Indexed:")
    print(indexed)
    npstring = np.array(indexed, np.int32).tostring()
    print("As String:")
    print(npstring)
    print("Length:")
    print(len(indexed))
    if len(indexed) >= hp.max_N:
        print("String is too long")
