#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts

Modified by sean leary. learysean1@hotmail.com
https://github.com/SeanPLeary/dc_tts-transfer-learning
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab, load_phns
from hyperparams import Hyperparams as hp
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *
import sys

import argparse


class Graph:
    def __init__(self, num=1, mode="train", datadir=hp.data, train_all=False):
        '''
        Args:
          num: Either 1 or 2. 1 for Text2Mel 2 for SSRN.
          mode: Either "train" or "synthesize".
        '''
        # Load vocabulary
        # self.char2idx, self.idx2char = load_vocab()
        # Load phonemes
        # self.phn2idx, self.idx2phn = load_phns()

        # Set flag
        training = True if mode == "train" else False

        # Graph
        # Data Feeding
        # L: Text. (B, N), int32
        # mels: Reduced melspectrogram. (B, T/r, n_mels) float32
        # mags: Magnitude. (B, T, n_fft//2+1) float32
        if mode == "train":
            self.L, self.mels, self.mags, self.fnames, self.num_batch = get_batch(
                datadir)
            self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
            self.gts = tf.convert_to_tensor(guided_attention())
        else:  # Synthesize
            self.L = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(
                tf.float32, shape=(None, None, hp.n_mels))
            self.prev_max_attentions = tf.placeholder(tf.int32, shape=(None,))

        if num == 1 or (not training):
            with tf.variable_scope("Text2Mel"):
                # Get S or decoder inputs. (B, T//r, n_mels)
                self.S = tf.concat(
                    (tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)

                # Networks
                with tf.variable_scope("TextEnc"):
                    self.K, self.V = TextEnc(
                        self.L, training=training)  # (N, Tx, e)

                with tf.variable_scope("AudioEnc"):
                    self.Q = AudioEnc(self.S, training=training)

                with tf.variable_scope("Attention"):
                    # R: (B, T/r, 2d)
                    # alignments: (B, N, T/r)
                    # max_attentions: (B,)
                    self.R, self.alignments, self.max_attentions = Attention(self.Q, self.K, self.V,
                                                                             mononotic_attention=(
                                                                                 not training),
                                                                             prev_max_attentions=self.prev_max_attentions)
                with tf.variable_scope("AudioDec"):
                    self.Y_logits, self.Y = AudioDec(
                        self.R, training=training)  # (B, T/r, n_mels)
        else:  # num==2 & training. Note that during training,
            # the ground truth melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(self.mels, training=training)

        if not training:
            # During inference, the predicted melspectrogram values are fed.
            with tf.variable_scope("SSRN"):
                self.Z_logits, self.Z = SSRN(self.Y, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(
                0, name='global_step', trainable=False)

        if training:
            if num == 1:  # Text2Mel
                # mel L1 loss
                self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))

                # mel binary divergence loss
                self.loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.Y_logits, labels=self.mels))

                # guided_attention loss
                self.A = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (
                    0, hp.max_T)], mode="CONSTANT", constant_values=-1.)[:, :hp.max_N, :hp.max_T]
                self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
                self.loss_att = tf.reduce_sum(
                    tf.abs(self.A * self.gts) * self.attention_masks)
                self.mask_sum = tf.reduce_sum(self.attention_masks)
                self.loss_att /= self.mask_sum

                # total loss
                self.loss = self.loss_mels + self.loss_bd1 + self.loss_att

                tf.summary.scalar('train/loss_mels', self.loss_mels)
                tf.summary.scalar('train/loss_bd1', self.loss_bd1)
                tf.summary.scalar('train/loss_att', self.loss_att)
                tf.summary.image(
                    'train/mel_gt', tf.expand_dims(tf.transpose(self.mels[:1], [0, 2, 1]), -1))
                tf.summary.image(
                    'train/mel_hat', tf.expand_dims(tf.transpose(self.Y[:1], [0, 2, 1]), -1))
            else:  # SSRN
                # mag L1 loss
                self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))

                # mag binary divergence loss
                self.loss_bd2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.Z_logits, labels=self.mags))

                # total loss
                self.loss = self.loss_mags + self.loss_bd2

                tf.summary.scalar('train/loss_mags', self.loss_mags)
                tf.summary.scalar('train/loss_bd2', self.loss_bd2)
                tf.summary.image(
                    'train/mag_gt', tf.expand_dims(tf.transpose(self.mags[:1], [0, 2, 1]), -1))
                tf.summary.image(
                    'train/mag_hat', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))

            # Training Scheme
            self.lr = learning_rate_decay(hp.lr, self.global_step)
            tvars = tf.trainable_variables()

            tvars_new = []
            for tvar in hp.selected_tvars:
                tvars_new = tvars_new + \
                    [var for var in tvars if tvar in var.name]

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar("lr", self.lr)

            # gradient clipping
            if not train_all:
                self.gvs = self.optimizer.compute_gradients(
                    self.loss, var_list=tvars_new)
            else:
                self.gvs = self.optimizer.compute_gradients(self.loss)

            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -1., 1.)
                self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(
                    self.clipped, global_step=self.global_step)

            # Summary
            self.merged = tf.summary.merge_all()


def get_arguments():
    parser = argparse.ArgumentParser(description='DC_TTS model trainer')
    parser.add_argument('num', type=int, help='Which model to train')
    parser.add_argument('--data', type=str, required=False,
                        help='Directory containing training data')
    parser.add_argument('--restore', type=str, required=False,
                        help='Directory containing pre-trained models')
    parser.add_argument('--new', action='store_true',
                        help="Don't load a previously trained model")
    parser.add_argument('--all', action='store_true', help="Train all layers")
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    # argument: 1 or 2. 1 for Text2mel, 2 for SSRN.
    num = int(args.num)

    if not num == 1 and not num == 2:
        print('Model argument must be 1 or 2')
        sys.exit()

    if args.data:
        if not os.path.exists(args.data):
            print('Directory %s not found. Exiting.' % args.data)
            sys.exit()
        else:
            datadir = args.data
    else:
        datadir = hp.data

    if args.restore:
        if not os.path.exists(args.restore):
            print('Directory %s not found. Exiting.' % args.restore)
            sys.exit()
        else:
            restoredir = args.restore + hp.logdir
    else:
        restoredir = hp.restoredir + hp.logdir

    g = Graph(num=num, datadir=datadir, train_all=args.all)
    print("Training Graph loaded")

    logdir = datadir + hp.logdir + "-" + str(num)
    sv = tf.train.Supervisor(
        logdir=logdir, save_model_secs=0, global_step=g.global_step)
    # with sv.managed_session() as sess:
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if not args.new:
            sv.saver.restore(sess, tf.train.latest_checkpoint(
                restoredir + "-" + str(num)))
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(
                        sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                    if num == 1:
                        # plot alignment
                        alignments = sess.run(g.alignments)
                        plot_alignment(alignments[0], str(
                            gs // 1000).zfill(3) + "k", logdir)

            # break
            if gs > hp.num_iterations:
                break

    print("Done")
