'''
Model for sequence to sequence learning. The model learns to generate a question given an answer,
and generalizes to other questions and answers.
Use KL-distance based adaptation techniques to adapt the baseline models
The adaptation data is actually copied from the dev data.
The paper for adaptation is
KL-divergence regularized deep neural network adaptation for improved large vocabulary speech recognition
'''

from __future__ import print_function
import numpy as np

import os
from keras.engine import Input
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Activation, Masking, merge, activations, Lambda, \
    ActivityRegularization
from keras.models import Model
import keras.backend as K
from keras.optimizers import RMSprop
import random

# can remove this depending on ide...
os.environ['INSURANCE_QA'] = 'c:/data/insurance_qa_python'

import sys

try:
    import cPickle as pickle
except:
    import pickle

model_save = 'models/answer_to_question.h5'
N_HIDDEN=128

class InsuranceQA:
    def __init__(self):
        try:
            data_path = os.environ['INSURANCE_QA']
        except KeyError:
            print("INSURANCE_QA is not set.  Set it to your clone of https://github.com/codekansas/insurance_qa_python")
            sys.exit(1)
        self.path = data_path
        self.vocab = self.load('vocabulary')
        self.table = InsuranceQA.VocabularyTable(self.vocab.values())

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    class VocabularyTable:
        ''' Identical to CharacterTable from Keras example '''
        def __init__(self, words):
            self.words = sorted(set(words))
            self.words_indices = dict((c, i) for i, c in enumerate(self.words))
            self.indices_words = dict((i, c) for i, c in enumerate(self.words))

        def encode(self, sentence, maxlen):
            indices = np.zeros((maxlen, len(self.words)))
            for i, w in enumerate(sentence):
                if i == maxlen: break
                indices[i, self.words_indices[w]] = 1
            return indices

        def decode(self, indices, calc_argmax=True):
            if calc_argmax:
                indices = np.argmax(indices, axis=-1)
                # indices = [self.sample(i) for i in indices]
            return ' '.join(self.indices_words[x] for x in indices)

        def sample(self, index, noise=0.2):
            index = np.log(index) / noise
            index = np.exp(index) / np.sum(np.exp(index))
            index = np.argmax(np.random.multinomial(1, index, 1))
            return index

class Evaluator:
    def __init__(self, conf=None):
        try:
            data_path = os.environ['INSURANCE_QA']
        except KeyError:
            print("INSURANCE_QA is not set.  Set it to your clone of https://github.com/codekansas/insurance_qa_python")
            sys.exit(1)
        self.path = data_path
        self.conf = dict() if conf is None else conf
        self.params = conf.get('training_params', dict())
        self.answers = self.load('answers')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None

    ##### Resources #####

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    def vocab(self):
        if self._vocab is None:
            self._vocab = self.load('vocabulary')
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            vocab = self.vocab()
            self._reverse_vocab = dict((v.lower(), k) for k, v in vocab.items())
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self, model, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        model.save_weights('models/weights_epoch_%d.h5' % epoch, overwrite=True)

    def load_epoch(self, model, epoch):
        assert os.path.exists('models/weights_epoch_%d.h5' % epoch), 'Weights at epoch %d not found' % epoch
        model.load_weights('models/weights_epoch_%d.h5' % epoch)

    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.get('question_len', None))

    def pada(self, data):
        return self.pad(data, self.conf.get('answer_len', None))

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def print_time(self):
        print(strftime('%Y-%m-%d %H:%M:%S :: ', gmtime()), end='')

    ##### Evaluation #####

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='')
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='')

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])
        return self._eval_sets

    def get_mrr(self, model, evaluate_all=False):
        top1s = list()
        mrrs = list()

        for name, data in self.eval_sets().items():
            if evaluate_all:
                self.print_time()
                print('----- %s -----' % name)

            random.shuffle(data)

            if not evaluate_all and 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            c_1, c_2 = 0, 0

            for i, d in enumerate(data):
                if evaluate_all:
                    self.prog_bar(i, len(data))

#                indices = d['good'] + d['bad']
                indices = d['good']
                answers = self.pada([self.answers[i] for i in indices])
                question = self.padq([d['question']] * len(indices))

                n_good = len(d['good'])
                sims = model.predict([question], batch_size=500).flatten()
                r = rankdata(sims, method='max')

                max_r = np.argmax(r)
                max_n = np.argmax(r[:n_good])

                # print(' '.join(self.revert(d['question'])))
                # print(' '.join(self.revert(self.answers[indices[max_r]])))
                # print(' '.join(self.revert(self.answers[indices[max_n]])))

                c_1 += 1 if max_r == max_n else 0
                c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            top1 = c_1 / float(len(data))
            mrr = c_2 / float(len(data))

            del data

            if evaluate_all:
                print('Top-1 Precision: %f' % top1)
                print('MRR: %f' % mrr)

            top1s.append(top1)
            mrrs.append(mrr)

        # rerun the evaluation if above some threshold
        if not evaluate_all:
            print('Top-1 Precision: {}'.format(top1s))
            print('MRR: {}'.format(mrrs))
            evaluate_all_threshold = self.params.get('evaluate_all_threshold', dict())
            evaluate_mode = evaluate_all_threshold.get('mode', 'all')
            mrr_theshold = evaluate_all_threshold.get('mrr', 1)
            top1_threshold = evaluate_all_threshold.get('top1', 1)

            if evaluate_mode == 'any':
                evaluate_all = evaluate_all or any([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or any([x >= mrr_theshold for x in mrrs])
            else:
                evaluate_all = evaluate_all or all([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or all([x >= mrr_theshold for x in mrrs])

            if evaluate_all:
                return self.get_mrr(model, evaluate_all=True)

        return top1s, mrrs

def get_model_for_adaptation(question_maxlen, answer_maxlen, vocab_len, n_hidden, learning_rate):
    question = Input(shape=(question_maxlen, vocab_len))
    masked = Masking(mask_value=0.)(question)

    # encoder rnn
    encode_rnn = LSTM(n_hidden, return_sequences=True, dropout_U=0.2)(masked)
    encode_rnn = LSTM(n_hidden, return_sequences=False, dropout_U=0.2)(encode_rnn)

    encode_brnn = LSTM(n_hidden, return_sequences=True, go_backwards=True, dropout_U=0.2)(masked)
    encode_brnn = LSTM(n_hidden, return_sequences=False, go_backwards=True, dropout_U=0.2)(encode_brnn)

    # repeat it maxlen times
    repeat_encoding_rnn = RepeatVector(answer_maxlen)(encode_rnn)
    repeat_encoding_brnn = RepeatVector(answer_maxlen)(encode_brnn)

    # decoder rnn
    decode_rnn = LSTM(n_hidden, return_sequences=True, dropout_U=0.2, dropout_W=0.5)(repeat_encoding_rnn)
    decode_rnn = LSTM(n_hidden, return_sequences=True, dropout_U=0.2)(decode_rnn)

    decode_brnn = LSTM(n_hidden, return_sequences=True, go_backwards=True, dropout_U=0.2, dropout_W=0.5)(repeat_encoding_brnn)
    decode_brnn = LSTM(n_hidden, return_sequences=True, go_backwards=True, dropout_U=0.2)(decode_brnn)

    merged_output = merge([decode_rnn, decode_brnn], mode='concat', concat_axis=-1)

    # output
    dense = TimeDistributed(Dense(vocab_len))(merged_output)
    regularized = ActivityRegularization(l2=1)(dense)
    softmax = Activation('softmax',name='softmax_node')(regularized)

    # compile the prediction model
    rmsprop = RMSprop(lr = learning_rate)
    model = Model([question], [softmax])
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop, metrics=['accuracy'])

    if os.path.exists(model_save):
        model.load_weights(model_save)
    else:
        print(model_save + ' not exists')
        raise Exception(model_save + ' not exists')
    return model

if __name__ == '__main__':
    question_maxlen, answer_maxlen = 20, 60

    conf = {
        'question_len': 30,
        'answer_len': 200,
        'n_words': 22353, # len(vocabulary) + 1
        'margin': 0.05,

        'training_params': {
            'save_every': 1,
            # 'eval_every': 1,
            'batch_size': 128,
            'nb_epoch': 1000,
            'validation_split': 0.2,
            'optimizer': 'adam',
            # 'optimizer': Adam(clip_norm=0.1),
            # 'n_eval': 100,

            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
        },

        'model_params': {
            'n_embed_dims': 100,
            'n_hidden': N_HIDDEN,

            # convolution
            'nb_filters': 1000, # * 4
            'conv_activation': 'relu',

            # recurrent
            'n_lstm_dims': 141, # * 2
        },

        'similarity_params': {
            'mode': 'gesd',
            'gamma': 1,
            'c': 1,
            'd': 2,
        }
    }

    qa = InsuranceQA()
    batch_size = 50
    n_test = 5
    kld_weight = 0.6  # the weight to the baseline model prediction. the larger the smooth/slow the adaptation will be

    print('Generating data...')
    answers = qa.load('answers')

    def gen_questions(batch_size, test=False):
        if test:
            questions = qa.load('test1')
        else:
            questions = qa.load('train')
        while True:
            i = 0
            question_idx = np.zeros(shape=(batch_size, question_maxlen, len(qa.vocab)))
            answer_idx = np.zeros(shape=(batch_size, answer_maxlen, len(qa.vocab)))
            for s in questions:
                if test:
                    ans = s['good']
                else:
                    ans = s['answers']
                for a in ans:
                    answer = qa.table.encode([qa.vocab[x] for x in answers[a]], answer_maxlen)
                    question = qa.table.encode([qa.vocab[x] for x in s['question']], question_maxlen)
                    # question = np.amax(question, axis=0, keepdims=False)
                    answer_idx[i] = answer
                    question_idx[i] = question
                    i += 1
                    if i == batch_size:
                        yield ([question_idx], [answer_idx])
                        i = 0

    def gen_adaptation_questions(batch_size):
        questions = qa.load('train')
        while True:
            i = 0
            question_idx = np.zeros(shape=(batch_size, question_maxlen, len(qa.vocab)))
            answer_idx = np.zeros(shape=(batch_size, answer_maxlen, len(qa.vocab)))
            for s in questions:
                ans = s['answers']
                for a in ans:
                    answer = qa.table.encode([qa.vocab[x] for x in answers[a]], answer_maxlen)
                    question = qa.table.encode([qa.vocab[x] for x in s['question']], question_maxlen)
                    # question = np.amax(question, axis=0, keepdims=False)
                    answer_idx[i] = answer
                    question_idx[i] = question
                    i += 1
                    if i == batch_size:
                        yield ([question_idx], [answer_idx])
                        i = 0

    def kld_adaptation(question_maxlen, answer_maxlen, qa, kld_weight, evaluator):
        print('Loaded trained model for adaptation...')
        model = get_model_for_adaptation(question_maxlen=question_maxlen, answer_maxlen=answer_maxlen, vocab_len=len(qa.vocab), n_hidden=N_HIDDEN, learning_rate=0.0001)

        print('Training model...')
        ix, iy = next(gen)
        for iteration in range(1, 20):  # original code runs 200 iterations
            print()
            print('-' * 50)
            print('Iteration', iteration)

            #check baseline
            evaluator.get_mrr(model)

            iz = model.predict(ix, )
            # use the original model to generate its prediction of the target distribution
            it = (1 - kld_weight) * iy[0] + kld_weight * iz
            # interpolate with the original target distribution
            model.fit(ix, [it], nb_epoch=10)
            model.save_weights(model_save + ".iter." + str(iteration), overwrite=True)

            print('Predict using the adapted model')
            x, y = next(test_gen)
            pred = model.predict(x, verbose=0)
            y = y[0]
            x = x[0]
            for i in range(n_test):
                print('Answer: {}'.format(qa.table.decode(x[i])))
                print('  Expected: {}'.format(qa.table.decode(y[i])))
                print('  Predicted: {}'.format(qa.table.decode(pred[i])))

            # generate new data for adaptation
            ix, iy = next(gen)

            evaluator.get_mrr(model)

    gen = gen_adaptation_questions(batch_size)
    test_gen = gen_questions(n_test, test=True)

    evaluator = Evaluator(conf)

    kld_adaptation(question_maxlen, answer_maxlen, qa, kld_weight, evaluator)

