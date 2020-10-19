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

import os


class Hyperparams:
    '''Hyper parameters'''
    # pipeline
    # if True, run `python prepro.py` first before running `python train.py`.
    prepro = True

    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = 0.97
    max_db = 100
    ref_db = 20

    # Model
    r = 4  # Reduction factor. Do not change this.
    dropout_rate = 0.1
    e = 128  # == embedding
    d = 256  # == hidden units of Text2Mel
    c = 512  # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = os.path.expanduser("~/voices/LJSpeech")
    test_data = 'test_sentences.txt'
    alphabet = "abcdefghijklmnopqrstuvwxyz'"
    punctuation = " ,.!?"
    # vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding, E: EOS.
    vocab = alphabet + punctuation
    # phonemes found in g2p_en
    phonemes = ['PAD', 'EOS'] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
                                 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
                                 'B', 'CH', 'D', 'DH',
                                 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
                                 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH',
                                 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
                                 'P', 'R', 'S', 'SH', 'T', 'TH',
                                 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2',
                                 'V', 'W', 'Y', 'Z', 'ZH'] + list(punctuation)
    max_N = 150  # Maximum number of characters.
    max_T = 210  # Maximum number of mel frames.

    # training scheme
    #lr = 0.001  # Initial learning rate.
    lr = 0.001
    logdir = "/logdir"
    meldir = "/mels"
    magdir = "/mags"
    restoredir = os.path.expanduser("~/voices/LJSpeech")
    sampledir = "/samples"
    B = 8
    num_iterations = 1200000
    #num_iterations = 1800000

    # select the trainable layers for transfer learning (i.e. remove the layers you want to fix during transfer learning)
    selected_tvars = [
        #'SSRN/C_1/',
        #'SSRN/HC_2/',
        #'SSRN/HC_3/',
        #'SSRN/D_4/',
        #'SSRN/HC_5/',
        #'SSRN/HC_6/',
        #'SSRN/D_7/',
        #'SSRN/HC_8/',
        #'SSRN/HC_9/',
        #'SSRN/C_10/',
        #'SSRN/HC_11/',
        #'SSRN/HC_12/',
        'SSRN/C_13/',
        'SSRN/C_14/',
        'SSRN/C_15/',
        'SSRN/C_16/',
        #'Text2Mel/TextEnc/embed_1/',
        #'Text2Mel/TextEnc/C_2/',
        #'Text2Mel/TextEnc/C_3/',
        #'Text2Mel/TextEnc/HC_4/',
        #'Text2Mel/TextEnc/HC_5/',
        #'Text2Mel/TextEnc/HC_6/',
        #'Text2Mel/TextEnc/HC_7/',
        #'Text2Mel/TextEnc/HC_8/',
        #'Text2Mel/TextEnc/HC_9/',
        #'Text2Mel/TextEnc/HC_10/',
        'Text2Mel/TextEnc/HC_11/',
        'Text2Mel/TextEnc/HC_12/',
        'Text2Mel/TextEnc/HC_13/',
        'Text2Mel/TextEnc/HC_14/',
        'Text2Mel/TextEnc/HC_15/',
        #'Text2Mel/AudioEnc/C_1/',
        #'Text2Mel/AudioEnc/C_2/',
        #'Text2Mel/AudioEnc/C_3/',
        #'Text2Mel/AudioEnc/HC_4/',
        #'Text2Mel/AudioEnc/HC_5/',
        #'Text2Mel/AudioEnc/HC_6/',
        #'Text2Mel/AudioEnc/HC_7/',
        #'Text2Mel/AudioEnc/HC_8/',
        'Text2Mel/AudioEnc/HC_9/',
        'Text2Mel/AudioEnc/HC_10/',
        'Text2Mel/AudioEnc/HC_11/',
        'Text2Mel/AudioEnc/HC_12/',
        'Text2Mel/AudioEnc/HC_13/',
        #'Text2Mel/AudioDec/C_1/',
        #'Text2Mel/AudioDec/HC_2/',
        #'Text2Mel/AudioDec/HC_3/',
        #'Text2Mel/AudioDec/HC_4/',
        #'Text2Mel/AudioDec/HC_5/',
        #'Text2Mel/AudioDec/HC_6/',
        'Text2Mel/AudioDec/HC_7/',
        'Text2Mel/AudioDec/C_8/',
        'Text2Mel/AudioDec/C_9/',
        'Text2Mel/AudioDec/C_10/',
        'Text2Mel/AudioDec/C_11/'
    ]
