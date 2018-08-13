import functools

import dynet as dy
import numpy as np

UNK = "<UNK>"


class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(
                self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x


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


class SingleHeadSelfAttentive(object):
    def __init__(
        self,
        model,
        d_model,
        d_k,
        d_v,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("SingleHeadSelfAttentive")

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = self.model.add_parameters((d_model, d_k))
        self.W_K = self.model.add_parameters((d_model, d_k))
        self.W_V = self.model.add_parameters((d_model, d_v))
        self.W_O = self.model.add_parameters((d_v, d_model))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, X):
        W_Q = self.W_Q.expr()
        W_K = self.W_K.expr()
        W_V = self.W_V.expr()
        W_O = self.W_O.expr()
        Q = X * W_Q
        K = X * W_K
        V = X * W_V
        QK = Q * dy.transpose(K) / np.sqrt(self.d_k)
        QKV = dy.softmax(QK, 1) * V * W_O
        return QKV


class MultiHeadSelfAttentive(object):
    def __init__(
        self,
        model,
        d_model,
        d_k,
        d_v,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("MultiHeadSelfAttentive")
        self.d_model = d_model

        self.feedforward = Feedforward(
            self.model, self.d_model, [self.d_model], self.d_model)
        self.attention = []
        for i in range(8):
            self.attention.append(SingleHeadSelfAttentive(
                self.model, d_model, d_k, d_v))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, X):
        d_x = X.dim()[0][0]
        d_y = X.dim()[0][1]
        g = dy.ones((d_x, d_y))
        b = dy.zeros((d_x, d_y))
        Y = []
        for attention in self.attention:
            Y.append(attention(X))
        Y = dy.esum(Y)
        Y = dy.layer_norm(X + Y, g, b)
        Y = dy.layer_norm(
            Y + dy.transpose(self.feedforward(dy.transpose(Y))), g, b)
        return Y


class AttentionEncoder(object):
    def __init__(
        self,
        model,
        d_model,
        d_k,
        d_v,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("AttentionEncoder")

        self.attention = []
        for i in range(8):
            self.attention.append(MultiHeadSelfAttentive(
                self.model, d_model, d_k, d_v))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, X):
        for attention in self.attention:
            X = attention(X)
        return X


class BiLSTMTagger(object):
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

        self.model = model.add_subcollection("BiLSTMTagger")
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
            self.model,
            2 * lstm_dim,
            label_hidden_dim,
            self.tag_vocab.size)

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


class AttentionTagger(object):
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
            pos_embedding_dim,
            max_sent_len,
            label_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("AttentionTagger")
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.char_lstm_dim = char_lstm_dim

        self.char_embeddings = self.model.add_lookup_parameters(
            (char_vocab.size, char_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))
        self.pos_embeddings = self.model.add_lookup_parameters(
            (max_sent_len, pos_embedding_dim))

        self.char_lstm = dy.BiRNNBuilder(
            char_lstm_layers,
            char_embedding_dim,
            2 * char_lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.attention = AttentionEncoder(
            self.model,
            2 * char_lstm_dim + word_embedding_dim + pos_embedding_dim,
            2 * char_lstm_dim + word_embedding_dim + pos_embedding_dim,
            2 * char_lstm_dim + word_embedding_dim + pos_embedding_dim)

        self.f_label = OutputLayer(
            self.model,
            2 * char_lstm_dim + word_embedding_dim + pos_embedding_dim,
            label_hidden_dim,
            self.tag_vocab.size)

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
        for pos, word in enumerate(words):
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
            pos_embedding = self.pos_embeddings[pos]
            embeddings.append(dy.concatenate(
                [char_embedding, word_embedding, pos_embedding]))

        embeddings = [dy.noise(e, 0.1) for e in embeddings]
        return embeddings

    def predict(self, words, gold=None):
        is_train = gold is not None

        embeddings = self.get_embeddings(words, is_train)
        embeddings = dy.concatenate(embeddings, 1)
        embeddings = dy.transpose(embeddings)
        attention_outputs = self.attention(embeddings)

        def helper():
            label_scores = []
            for idx in range(attention_outputs.dim()[0][0]):
                x = attention_outputs[idx]
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
