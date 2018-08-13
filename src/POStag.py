import functools

import dynet as dy
import numpy as np

UNK = "<UNK>"


class OutputLayer(object):
    def __init__(self, model, input_dim, hidden_dim, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("OutputLayer")

        self.H = model.add_parameters((hidden_dim, input_dim))
        self.O = model.add_parameters((output_dim, hidden_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        H = self.H.expr()
        O = self.O.expr()
        y = O * (dy.tanh(H * x))
        return y


class POSTagger(object):
    def __init__(
            self,
            model,
            tag_vocab,
            char_vocab,
            word_vocab,
            char_embedding_dim,
            char_lstm_layers,
            char_lstm_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("POSTagger")
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.char_lstm_dim = char_lstm_dim
        self.lstm_dim = lstm_dim

        self.char_embeddings = self.model.add_lookup_parameters(
            (char_vocab.size, char_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.char_lstm = dy.BiRNNBuilder(
            char_lstm_layers,
            char_embedding_dim,
            2 * char_lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            2 * char_lstm_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = OutputLayer(
            self.model, 2 * lstm_dim, label_hidden_dim, self.tag_vocab.size)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def get_embeddings(self, words, is_train):
        if is_train:
            self.char_lstm.set_dropout(self.dropout)
        else:
            self.char_lstm.disable_dropout()

        embeddings = []
        for word in words:
            count = self.word_vocab.count(word)
            if not count:
                word = UNK

            chars = list(word)
            char_lstm_outputs = self.char_lstm.transduce([
                self.char_embeddings[self.char_vocab.index_or_unk(char, UNK)]
                for char in chars])
            char_embedding = dy.concatenate([
                char_lstm_outputs[-1][:self.char_lstm_dim],
                char_lstm_outputs[0][self.char_lstm_dim:]])

            word_embedding = self.word_embeddings[self.word_vocab.index(word)]

            embeddings.append(dy.concatenate([char_embedding, word_embedding]))

        embeddings = [dy.noise(e, 0.1) for e in embeddings]
        return embeddings

    def predict(self, words, gold=None):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = self.get_embeddings(words, is_train)

        lstm_outputs = self.lstm.transduce(embeddings)

        def helper():
            label_scores = []
            for x in lstm_outputs:
                label_score = self.f_label(x)
                label_scores.append(label_score)
            total_loss = dy.zeros(1)
            tags = []
            if is_train:
                losses = []
                for label_score, tag in zip(label_scores, gold):
                    tag_index = self.tag_vocab.index(tag)
                    loss = dy.pickneglogsoftmax(label_score, tag_index)
                    losses.append(loss)
                total_loss = dy.esum(losses)
            label_scores = [dy.softmax(ls) for ls in label_scores]
            probs = [ls.npvalue() for ls in label_scores]
            for prob in probs:
                tag_index = np.argmax(prob)
                tag = self.tag_vocab.value(tag_index)
                tags.append(tag)

            return tags, total_loss

        tags, loss = helper()
        return tags, loss
