import functools

import dynet as dy
import numpy as np
import utils
import time

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"


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
        self.trans_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_vocab.size))

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

        self.f_label = utils.OutputLayer(
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
        for word in [START] + words + [STOP]:
            count = self.word_vocab.count(word)
            if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                word = UNK

            chars = list(word) if word not in (START, STOP) else [word]
            char_lstm_outputs = self.char_lstm.transduce([
                self.char_embeddings[self.char_vocab.index_or_unk(char, UNK)]
                for char in [START] + chars + [STOP]])
            char_embedding = dy.concatenate([
                char_lstm_outputs[-1][:self.char_lstm_dim],
                char_lstm_outputs[0][self.char_lstm_dim:]])

            word_embedding = self.word_embeddings[self.word_vocab.index(word)]

            embeddings.append(dy.concatenate([char_embedding, word_embedding]))

        embeddings = [dy.noise(e, 0.1) for e in embeddings]
        return embeddings

    def predict(self, words, gold=None, use_crf=False):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = self.get_embeddings(words, is_train)

        lstm_outputs = self.lstm.transduce(embeddings)

        def viterbi_decoding(label_scores, gold=[]):
            init_prob = [-np.inf] * self.tag_vocab.size
            init_prob[self.tag_vocab.index(START)] = 0
            forward_expr = dy.inputVector(init_prob)
            best_idxs = []
            trans_exprs = [self.trans_embeddings[idx]
                           for idx in range(self.tag_vocab.size)]

            for i, label_score in enumerate(label_scores):
                forward_best_idxs = []
                forward_best_exprs = []
                for next_tag in range(self.tag_vocab.size):
                    next_single_expr = forward_expr + trans_exprs[next_tag]
                    next_single = next_single_expr.npvalue()
                    forward_best_idx = np.argmax(next_single)
                    forward_best_idxs.append(forward_best_idx)
                    forward_best_exprs.append(
                        dy.pick(next_single_expr, forward_best_idx))

                forward_expr = dy.concatenate(forward_best_exprs) + label_score
                best_idxs.append(forward_best_idxs)

            next_single_expr = forward_expr + \
                trans_exprs[self.tag_vocab.index(STOP)]
            next_single = next_single_expr.npvalue()
            forward_best_idx = np.argmax(next_single)
            best_expr = dy.pick(next_single_expr, forward_best_idx)

            best_path = [self.tag_vocab.value(forward_best_idx)]
            for forward_best_idxs in reversed(best_idxs):
                forward_best_idx = forward_best_idxs[forward_best_idx]
                best_path.append(self.tag_vocab.value(forward_best_idx))
            best_path.pop()
            best_path.reverse()

            return best_path, best_expr

        def forced_decoding(label_scores, gold):
            forward_expr = dy.zeros(1)
            forward_tag_idx = self.tag_vocab.index(START)
            trans_exprs = [self.trans_embeddings[idx]
                           for idx in range(self.tag_vocab.size)]

            for i, label_score in enumerate(label_scores):
                next_tag_idx = self.tag_vocab.index(gold[i])
                forward_expr = forward_expr + \
                    dy.pick(trans_exprs[next_tag_idx],
                            forward_tag_idx) + label_score[next_tag_idx]
                forward_tag_idx = next_tag_idx
            forward_expr = forward_expr + \
                dy.pick(
                    trans_exprs[self.tag_vocab.index(STOP)], forward_tag_idx)

            return forward_expr

        def helper():
            label_scores = []
            for x in lstm_outputs[1:-1]:
                label_score = self.f_label(x)
                label_scores.append(label_score)

            if use_crf:
                tags, viterbi_scores = viterbi_decoding(label_scores, gold)
                if is_train and tags != gold:
                    gold_scores = forced_decoding(label_scores, gold)
                    total_loss = viterbi_scores - gold_scores
                else:
                    total_loss = dy.zeros(1)
            else:
                total_loss = dy.zeros(1)
                tags = []
                if is_train:
                    losses = []
                    for label_score, tag in zip(label_scores, gold):
                        tag_index = self.tag_vocab.index(tag)
                        loss = dy.pickneglogsoftmax(label_score, tag_index)
                        losses.append(loss)
                    total_loss = dy.esum(losses)
                else:
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

        self.attention = utils.SingleHeadSelfAttentive(
            self.model,
            2 * char_lstm_dim + word_embedding_dim + pos_embedding_dim,
            2 * char_lstm_dim + word_embedding_dim + pos_embedding_dim,
            2 * char_lstm_dim + word_embedding_dim + pos_embedding_dim)

        self.f_label = utils.OutputLayer(
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
            else:
                label_scores = [dy.softmax(ls) for ls in label_scores]
                probs = [ls.npvalue() for ls in label_scores]
                for prob in probs:
                    tag_index = np.argmax(prob)
                    tag = self.tag_vocab.value(tag_index)
                    tags.append(tag)

            return tags, total_loss

        tags, loss = helper()
        return tags, loss
