import argparse
import itertools
import os.path
import time
import sys

import dynet as dy
import numpy as np

import POStag
import vocabulary


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def read(filename):
    with open(filename) as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("/", 1)) for x in line]
            yield sent


def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training sentences from {}...".format(args.train_path))
    train_sents = list(read(args.train_path))
    print("Loaded {:,} training examples.".format(len(train_sents)))

    print("Loading development sentences from {}...".format(args.dev_path))
    dev_sents = list(read(args.dev_path))
    print("Loaded {:,} development examples.".format(len(dev_sents)))

    print("Constructing vocabularies...")
    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(POStag.UNK)
    tag_vocab.index(POStag.START)
    tag_vocab.index(POStag.STOP)

    char_vocab = vocabulary.Vocabulary()
    char_vocab.index(POStag.UNK)
    char_vocab.index(POStag.START)
    char_vocab.index(POStag.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(POStag.UNK)
    char_vocab.index(POStag.START)
    char_vocab.index(POStag.STOP)

    for sent in train_sents:
        for word, tag in sent:
            tag_vocab.index(tag)
            word_vocab.index(word)
            for char in word:
                char_vocab.index(char)

    tag_vocab.freeze()
    char_vocab.freeze()
    word_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {POStag.UNK, POStag.START, POStag.STOP}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Character", char_vocab)
        print_vocabulary("Word", word_vocab)

    print("Initializing model...")

    model = dy.ParameterCollection()
    if args.tagger_type == "bilstm":
        tagger = POStag.BiLSTMTagger(
            model,
            tag_vocab,
            char_vocab,
            word_vocab,
            args.char_embedding_dim,
            args.char_lstm_layers,
            args.char_lstm_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    else:
        tagger = POStag.AttentionTagger(
            model,
            tag_vocab,
            char_vocab,
            word_vocab,
            args.char_embedding_dim,
            args.char_lstm_layers,
            args.char_lstm_dim,
            args.word_embedding_dim,
            args.pos_embedding_dim,
            args.max_sent_len,
            args.label_hidden_dim,
            args.dropout,
        )
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_sents) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None

    start_time = time.time()

    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        good_sent = bad_sent = good = bad = 0.0
        for sent in dev_sents:
            dy.renew_cg()
            words = [word for word, tag in sent]
            golds = [tag for word, tag in sent]
            predicted, loss = tagger.predict(words, None, args.use_crf)

            if predicted == golds:
                good_sent += 1
            else:
                bad_sent += 1
            for go, gu in zip(golds, predicted):
                if go == gu:
                    good += 1
                else:
                    bad += 1

        dev_fscore = good / (good + bad) * 100
        dev_sent_fscore = good_sent / (good_sent + bad_sent) * 100

        print(
            "\n"
            "dev-fscore {:.2f} "
            "dev-sent-fscore {:.2f} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                dev_sent_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time)
            )
        )

        if dev_fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Romving previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [tagger])

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_sents)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_sents), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for sent in train_sents[start_index:start_index + args.batch_size]:
                words = [word for word, tag in sent]
                tags = [tag for word, tag in sent]
                predicted, loss = tagger.predict(words, tags, args.use_crf)
                batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "\r"
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_sents) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time)
                ),
                end=""
            )
            sys.stdout.flush()

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()


def run_test(args):
    print("Loading test sentences from {}...".format(args.test_path))
    test_sents = list(read(args.test_path))
    print("Loaded {:,} test examples.".format(len(test_sents)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [tagger] = dy.load(args.model_path_base, model)

    print("Tagging test sentences...")
    test_start_time = time.time()
    good_sent = bad_sent = good = bad = 0.0

    for sent in test_sents:
        dy.renew_cg()
        words = [word for word, tag in sent]
        golds = [tag for word, tag in sent]

        predicted, loss = tagger.predict(words, None, args.use_crf)

        if predicted == golds:
            good_sent += 1
        else:
            bad_sent += 1
        for go, gu in zip(golds, predicted):
            if go == gu:
                good += 1
            else:
                bad += 1

    test_fscore = good / (good + bad) * 100
    test_sent_fscore = good_sent / (good_sent + bad_sent) * 100

    print(
        "\n"
        "test-fscore {:.2f} "
        "test-sent-fscore {:.2f} "
        "test-elapsed {} ".format(
            test_fscore,
            test_sent_fscore,
            format_elapsed(test_start_time)
        )
    )


def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument(
        "--tagger-type", choices=["bilstm", "attention"], required=True)
    subparser.add_argument("--char-embedding-dim", type=int, default=32)
    subparser.add_argument("--char-lstm-layers", type=int, default=1)
    subparser.add_argument("--char-lstm-dim", type=int, default=32)
    subparser.add_argument("--word-embedding-dim", type=int, default=256)
    subparser.add_argument("--pos-embedding-dim", type=int, default=256)
    subparser.add_argument("--max-sent-len", type=int, default=300)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--lstm-dim", type=int, default=64)
    subparser.add_argument("--label-hidden-dim", type=int, default=128)
    subparser.add_argument("--use-crf", action="store_true")
    subparser.add_argument("--dropout", type=float, default=0.1)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--train-path", default="data/train.data")
    subparser.add_argument("--dev-path", default="data/dev.data")
    subparser.add_argument("--batch-size", type=int, default=64)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--test-path", default="data/test.data")

    args = parser.parse_args()
    args.callback(args)


if __name__ == '__main__':
    main()
