# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts

Modified by sean leary. learysean1@hotmail.com
https://github.com/SeanPLeary/dc_tts-transfer-learning

Modified
https://github.com/kwmkwm/dc_tts-phonetic-transfer-learning
'''

from __future__ import print_function
from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
import sys
from g2p_en import G2p


g2p = G2p()


def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char


def load_phns():
    phn2idx = {phn: idx for idx, phn in enumerate(hp.phonemes)}
    idx2phn = {idx: phn for idx, phn in enumerate(hp.phonemes)}
    return phn2idx, idx2phn


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.replace('-', ' ')
    # pauses are pauses, right?
    text = text.replace(':', ',')
    text = text.replace(';', ',')
    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    text = text.strip()
    return text


def phoneme_normalize(phones,verbose=False):
    pre_processed_phones = []
    processed_phones = []
    # remove unwanted characters
    for phone in phones:
        if (phone in hp.phonemes):
            pre_processed_phones.append(phone)
        else:
            if verbose:
                print("dropping unexpected phone")
                print(phone)

    # G2p adds spaces before punctuation
    # lets remove them
    for idx, p_phone in enumerate(pre_processed_phones):
        if not p_phone == ' ':
            processed_phones.append(p_phone)
        else:
            if idx + 1 < len(pre_processed_phones):
                if not (pre_processed_phones[idx + 1] in hp.punctuation):
                    processed_phones.append(p_phone)
                else:
                    if verbose:
                        print("dropping unnecessary space")
            else:
                if verbose:
                    print("dropping dangling space")

    return processed_phones


def load_data(datadir, mode="train", txtfile=hp.test_data):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    #char2idx, idx2char = load_vocab()
    phn2idx, idx2phn = load_phns()

    if mode == "train":
        # Parse
        #fpaths, text_lengths, texts = [], [], []
        fpaths, phone_lengths, phones = [], [], []
        transcript = os.path.join(datadir, 'metadata.csv')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        for line in lines:
            #if "LJ" in datadir:
            fname, _, text = line.strip().split("|")
            #else:
            #    fname, text = line.strip().split("|")

            numbers = re.search('[0-9]+', text)
            test1 = re.search('&', text)

            if numbers is None and test1 is None:
                fpath = os.path.join(datadir, "wavs", fname + ".wav")
                fpaths.append(fpath)
                # text = text_normalize(text) + "E"  # E: EOS
                text = text_normalize(text)
                phone = g2p(text)
                phone = phoneme_normalize(phone) + ["EOS"]
                #text = [char2idx[char] for char in text]
                phone = [phn2idx[phn] for phn in phone]
                # text_lengths.append(len(text))
                phone_lengths.append(len(phone))
                #texts.append(np.array(text, np.int32).tostring())
                phones.append(np.array(phone, np.int32).tostring())

        # return fpaths, text_lengths, texts
        return fpaths, phone_lengths, phones

    else:  # synthesize on unseen test text.
        # Parse
        lines = codecs.open(txtfile, 'r', 'utf-8').readlines()
        #sents = [text_normalize(line) + "E" for line in lines]
        sents = [text_normalize(line) for line in lines]

        #texts = np.zeros((len(sents), hp.max_N), np.int32)
        # for i, sent in enumerate(sents):
        #    texts[i, :len(sent)] = [char2idx[char] for char in sent]

        psents = [g2p(sent) for sent in sents]
        pnorms = [phoneme_normalize(psent) + ["EOS"] for psent in psents]
        phones = np.zeros((len(pnorms), hp.max_N), np.int32)
        for i, pnorm in enumerate(pnorms):
            phones[i, :len(pnorm)] = [phn2idx[phn] for phn in pnorm]

        # return texts
        return phones


def get_batch(datadir):
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        # fpaths, text_lengths, texts = load_data(datadir)  # list
        fpaths, phone_lengths, phones = load_data(datadir)  # list
        #maxlen, minlen = max(text_lengths), min(text_lengths)
        maxlen, minlen = max(phone_lengths), min(phone_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        # fpath, text_length, text = tf.train.slice_input_producer(
        #    [fpaths, text_lengths, texts], shuffle=True)
        fpath, phone_length, phone = tf.train.slice_input_producer(
            [fpaths, phone_lengths, phones], shuffle=True)

        # Parse
        # text = tf.decode_raw(text, tf.int32)  # (None,)
        phone = tf.decode_raw(phone, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = datadir + hp.meldir + \
                    "/{}".format(fname.decode("utf-8").replace("wav", "npy"))
                mag = datadir + hp.magdir + \
                    "/{}".format(fname.decode("utf-8").replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [
                                         tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [
                                         tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        # text.set_shape((None,))
        phone.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft // 2 + 1))

        # Batching
        # _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
        #    input_length=text_length,
        #    tensors=[text, mel, mag, fname],
        _, (phones, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
            input_length=phone_length,
            tensors=[phone, mel, mag, fname],
            batch_size=hp.B,
            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
            num_threads=8,
            capacity=hp.B * 4,
            dynamic_pad=True)

    # return texts, mels, mags, fnames, num_batch
    return phones, mels, mags, fnames, num_batch
