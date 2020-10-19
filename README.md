# dc_tts-phonetic-transfer-learning

This repo contains attempts to improve the accuracy of [SeanPLeary's dc_tts-transfer-learning code](https://github.com/SeanPLeary/dc_tts-transfer-learning). This is done by first converting all text to a phonetic representation using [Kyubyong's g2p library](https://github.com/Kyubyong/g2p). The conversion is done automatically without invervention from the user. These modifications necessitate changes in the model structure that are not compatible with the original code, so new models are required. A model which has been trained using the LJSpeech dataset can be found [here](https://www.dropbox.com/s/415qb3mnnnmhwb0/LJSpeech-phonetic.tar?dl=0).

---

# Additional Changes

For consitency and reduced code, datasets must match the format of the LJSpeed dataset including using the filename "metadata.csv" rather than "transcript.csv".

Command line options have been added to make it easy to switch between models without editing the hyperparams.py file

---

# prepro.py

The "--data" option allows users to select the directory in which the metadata.csv file should be found. The "mags" and "mels" subdirectories will be created in that directory.

# train_transfer.py

The "1" and "2" options select which network to train. "1" for Text2Mel, "2" for SSRN.

The "--data" option opperates the same as with the prepro.py script.

The "--restore" option selects a directory containing a previously trained model. This can be useful if you need to interrupt training and start again later.

The "--new" option prevents the script from loading a previously trained model. If you use this option you should also use the "--all" option.

The "--all" option will train all layers, rather than only the layers selected in the hyperparams.py file.

# synthesize.py

The previous version of the code skipped the first line of the text file and the first word of each remaining line. I could see no reason for this behavior so I have changed it. Output filenames were previously indexed from 1, I have changed this to index from 0.

The "--voice" option selects a diretory containing a trained model, similar to the "--data" option of the train_transfer.py script.

The "--text" option selects a text file to read.

The "--outdir" option selects a directory to save the .wav files.

# normalize.py

This script accepts a text string. It will output the phonetic representation of that string. This can be useful when tracking down the source of pronounciation issues.

---

# Issues

The models are trained using only ASCII characters. Unrecognized characters are converted to spaces. This can cause problems if unicode characters are used. For example, an ASCII apostrophe will work as expected but a unicode U+2019 will not.

I have had difficulty training exceptionally deep voices. Using the "--all" option to train all layers has been helpful.

---

If you would like to train a model from scratch, the LJSpeech 1.1 dataset can be found [here](https://keithito.com/LJ-Speech-Dataset/). I recommend using a modified metadata.csv file found [here](https://github.com/kwmkwm/LJSpeech1.1-expanded).
