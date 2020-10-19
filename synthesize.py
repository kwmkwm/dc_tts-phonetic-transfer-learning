#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts

Modified by sean leary. learysean1@hotmail.com
https://github.com/SeanPLeary/dc_tts-transfer-learning

Modified
https://github.com/kwmkwm/dc_tts-phonetic-transfer-learning
'''

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train_transfer import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm

import argparse
import sys


def synthesize(logdir, txtfile, outdir):
    # Load data
    L = load_data("_",mode="synthesize", txtfile=txtfile)

    # Load graph
    g = Graph(mode="synthesize")
    print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        # mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for i, mag in enumerate(Z):
            print("Working on file", i)
            wav = spectrogram2wav(mag)
            write(outdir + "/" + os.path.basename(txtfile) +
                  ".{}.wav".format(i), hp.sr, wav)


def get_arguments():
    parser = argparse.ArgumentParser(description='DC_TTS synthesizer')
    parser.add_argument('--voice', type=str, required=False,
                        help='Directory containing output/logdir subdirectories')
    parser.add_argument('--text', type=str, required=False, help='TXT file')
    parser.add_argument('--outdir', type=str, required=False,
                        help='Directory to output wav files')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    if args.voice:
        if not os.path.exists(args.voice):
            print('Directory %s not found. Exiting.' % args.voice)
            sys.exit()
        else:
            voicedir = args.voice
    else:
        voicedir = hp.restoredir

    logdir = voicedir + hp.logdir

    if args.text:
        if not os.path.exists(args.text):
            print('File %s not found. Exiting.' % args.text)
            sys.exit()
        else:
            txtfile = args.text
    else:
        txtfile = hp.test_data

    if args.outdir:
        if not os.path.exists(args.outdir):
            print('Directory %s not found. Exiting.' % args.outdir)
            sys.exit()
        else:
            outdir = args.outdir
    else:
        outdir = voicedir + hp.sampledir

    synthesize(logdir, txtfile, outdir)
    print("Done")
