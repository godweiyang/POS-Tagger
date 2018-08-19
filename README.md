# Part-of-Speech Tagger
## Requirements and Setup

* Python 3.5 or higher.
* [DyNet](https://github.com/clab/dynet). We recommend installing DyNet from source with MKL support for significantly faster run time.

## Training

A new model can be trained using the command `python3 src/main.py train ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--numpy-seed` | NumPy random seed | Random
`--tagger-type` | `bilstm` or `attention` | N/A
`--char-embedding-dim` | Character embedding dimension |32
`--char-lstm-layers` | Number of bidirectional Char-LSTM layers |1
`--char-lstm-dim` | Hidden dimension of each Char-LSTM within each layer |32
`--word-embedding-dim` | Word embedding dimension | 256 
`--pos-embedding-dim` | Position embedding dimension | 256 
`--max-sent-len` | Maximum length of sentences | 300 
`--lstm-layers` | Number of bidirectional LSTM layers | 2
`--lstm-dim` | Hidden dimension of each LSTM within each layer | 64
`--label-hidden-dim` | Hidden dimension of label-scoring feedforward network | 128 
`--use-crf` | Whether or not to use CRF layer | Do not use the CRF layer 
`--dropout` | Dropout rate for LSTMs | 0.1 
`--model-path-base` | Path base to use for saving models | N/A
`--train-path` | Path to training sentences | `data/train.data`
`--dev-path` | Path to development sentences | `data/dev.data`
`--batch-size` | Number of examples per training update | 64 
`--epochs` | Number of training epochs | No limit
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--print-vocabs` | Print the vocabularies before training | Do not print the vocabularies

To train a tagger using the default hyperparameters, you can use the command:

```
python3 src/main.py train --tagger-type bilstm --model-path-base models/bilstm-model
```

## Evaluation

A saved model can be evaluated on a test corpus using the command `python3 src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--test-path` | Path to test sentences | `data/test.data`


As an example, after extracting the pre-trained bilstm model, you can evaluate it on the test set using the following command:

```
python3 src/main.py test --model-path-base models/bilstm-model_dev=xx.xx
```
