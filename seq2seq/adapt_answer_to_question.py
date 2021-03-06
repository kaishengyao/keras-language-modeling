'''
Model for sequence to sequence learning. The model learns to generate a question given an answer,
and generalizes to other questions and answers.
Use adaptation techniques to adapt the baseline models
The adaptation data is actually copied from the dev data.
To-Do: use true adaptation data
'''

from __future__ import print_function
import numpy as np
from scipy.stats import rankdata
import argparse
import os
from keras.engine import Input
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Activation, Masking, merge, activations, Lambda, \
    ActivityRegularization
from keras.models import Model
import keras.backend as K
from keras.optimizers import RMSprop

# can remove this depending on ide...
os.environ['INSURANCE_QA'] = 'c:/data/insurance_qa_python'

import sys

try:
    import cPickle as pickle
except:
    import pickle

model_save = 'models/answer_to_question.h5'

''' ====================================================== '''
''' the following is global setup for experimentation'''
kld_weight = 0.2  # weight to the original model, 1 corresponds to unsupervised training with target from the original model

''' adaptation technique
'simple' : simple adaptation using online adaptation
'kld'    : KL-distance based adpatation
'''
adaptation_technique = 'simple'

num_hypothesis = 10

''' ====================================================== '''

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
            slen = 0
            for i, w in enumerate(sentence):
                if i == maxlen: break
                indices[i, self.words_indices[w]] = 1
                slen += 1
            return indices, slen

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

def get_model_for_adaptation(model_name, question_maxlen, answer_maxlen, vocab_len, n_hidden, learning_rate, must_exist_before):
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
    softmax = Activation('softmax')(regularized)

    # compile the prediction model
    rmsprop = RMSprop(lr = learning_rate)
    model = Model([question], [softmax])
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    if os.path.exists(model_name) and must_exist_before:
        model.load_weights(model_name)
    elif not must_exist_before:
        return model
    else:
        print(model_name + ' not exists')
        raise Exception(model_name + ' not exists')
    return model

if __name__ == '__main__':

    def padq(data):
        return pad(data, question_maxlen)

    def pada(data):
        return pad(data, answer_maxlen)

    def pad(data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    '''
        get question, answer pairs
        answer include good and bad answers
        return 0) batch_size, 1) number of good answers, 2) list of questions, 3) answers including good and bad
    '''
    def gen_questions(question_answer):
        ans = question_answer['good'] + question_answer['bad']
        i = 0
        batch_size = len(ans)
        question_idx = np.zeros(shape=(batch_size, question_maxlen, len(qa.vocab)))
        answer_idx = np.zeros(shape=(batch_size, answer_maxlen, len(qa.vocab)))

        v_question_len = []
        v_answer_len = []
        for a in ans:
            answer, answer_len = qa.table.encode([qa.vocab[x] for x in answers[a]], answer_maxlen)
            question, question_len = qa.table.encode([qa.vocab[x] for x in question_answer['question']], question_maxlen)

            answer_idx[i] = answer
            question_idx[i] = question

            v_question_len.append(question_len)
            v_answer_len.append(answer_len)
            i += 1

        return batch_size, len(question_answer['good']), [question_idx], [answer_idx], v_answer_len, v_question_len

    def get_recall_rate(model):
        top1s = list()
        mrrs = list()


        questions = qa.load('test2')
        c_1, c_5, c_10 = 0, 0, 0

        datasize = 0
        for s in questions:
            bsize, n_good, question, lanswers, ans_len, qus_len = gen_questions(s)

            idx = 0
            sims = []
            for idx in xrange(min(num_hypothesis, bsize)): # 10 candidates including good responses
                q = question[0][idx,:,:]
                a = lanswers[0][idx,:,:]
                answer_idx = np.zeros(shape=(1, answer_maxlen, len(qa.vocab)))
                question_idx = np.zeros(shape=(1, question_maxlen, len(qa.vocab)))
                answer_idx[0] = a
                question_idx[0] = q
                #
                # sims.append(model.evaluate([question_idx], [answer_idx], batch_size=1)[0]) Theano's cross entropy backend. Not normalized with length of sentences
                #

                sims.append(model.evaluate([question_idx], [answer_idx], batch_size=1,verbose=0)[0] / float(ans_len[idx]))
                # normalize the cross-entropy w.r.t. the number of words in answer
            r = rankdata(sims, method='max')

            max_r = np.argmax(r)
            max_n = np.argmax(r[:n_good])

            c_1 += 1 if max_r < n_good else 0   #recall rate R@1
            c_5 += 1 if len(r) - r[max_n] < 5 else 0       #recall rate R@5
            c_10 += 1 if len(r) - r[max_n] < 10 else 0       #recall rate R@10

            '''
            c_1 += 1 if max_r < n_good else 0   #recall rate R@1
            c_2 += 1 if max_r < 10 else 0       #recall rate R@5
            c_2 += 1 if max_r < 10 else 0       #recall rate R@10
            '''
            datasize += 1

        R1 = c_1 / float(datasize)
        R5 = c_5 / float(datasize)
        R10 = c_10 / float(datasize)

        print('R@1 : %f' % R1)
        print('R@5 : %f' % R5)
        print('R@10 : %f' % R10)

        return R1, R5, R10


    def gen_adaptation_questions(batch_size):
#        questions = qa.load('train')
        # need to make sure that the adaptaiton data can be read and its format is the same as training
        # in my case, I don't find adaptation, so I just copy training data and use it for adaptation
        questions = qa.load('test1')
        while True:
            i = 0
            question_idx = np.zeros(shape=(batch_size, question_maxlen, len(qa.vocab)))
            answer_idx = np.zeros(shape=(batch_size, answer_maxlen, len(qa.vocab)))
            for s in questions:
                ans = s['good']
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

    def simple_adaptation(model, question_maxlen, answer_maxlen, qa):

        print('Training model...')
        for iteration in range(1, max_iteration):  # original code runs 200 iterations
            print()
            print('-' * 50)
            print('Iteration', iteration)

            model.fit_generator(gen, samples_per_epoch=10*batch_size, nb_epoch=1)
            model.save_weights(model_save + ".simple.adpt.iter." + str(iteration), overwrite=True)

            print('Evaluate performance')
            get_recall_rate(model)

    def kld_adaptation(model, question_maxlen, answer_maxlen, qa, kld_weight):

        print('Training model...')
        ix, iy = next(gen)
        for iteration in range(1, max_iteration):  # original code runs 200 iterations
            print()
            print('-' * 50)
            print('Iteration', iteration)

            iz = model.predict(ix, )
            # use the original model to generate its prediction of the target distribution
            it = (1 - kld_weight) * iy[0] + kld_weight * iz
            # interpolate with the original target distribution
            model.fit(ix, [it], nb_epoch=1)
            model.save_weights(model_save + ".kld.adpt.iter." + str(iteration), overwrite=True)

            get_recall_rate(model)

            # generate new data for adaptation
            ix, iy = next(gen)


    qa = InsuranceQA()
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptation_technique", help="adaptation techniques, 'simple' or 'kld' ", choices=["simple", "kld"], default="simple")
    parser.add_argument("--init_model", help="the initial model name", default=model_save)
    parser.add_argument("--batch_size", help="the batch (number of sentences to be processed in parallel) size", default=50)
    parser.add_argument("--question_maxlen", help="the maximum number of words in a question", default=20)
    parser.add_argument("--answer_maxlen", help="the maximum number of words in an answer", default=60)
    parser.add_argument("--kld_weight", help="the weight to the original model when doing KLD adpatation", default=0.2)
    parser.add_argument("--max_iteration", help="maximum numbre of adaptation", default=20)
    parser.add_argument("--learning_rate", help="learning rate for adaptation", default=0.0001)
    parser.add_argument("--num_hypothesis", help="number of hypothesis to evaluate recall rates", default=10)

    args = parser.parse_args()

    adaptation_technique = args.adaptation_technique
    init_model = args.init_model
    batch_size = args.batch_size
    kld_weight = args.kld_weight
    num_hypothesis = args.num_hypothesis

    question_maxlen, answer_maxlen = args.question_maxlen, args.answer_maxlen
    learning_rate = args.learning_rate
    max_iteration = args.max_iteration

    print('Generating data...')
    answers = qa.load('answers')

    gen = gen_adaptation_questions(batch_size)

    print('Loaded trained model for adaptation...')
    model = get_model_for_adaptation(model_name=init_model, question_maxlen=question_maxlen,
                                     answer_maxlen=answer_maxlen, vocab_len=len(qa.vocab), n_hidden=128,
                                     learning_rate=learning_rate, must_exist_before=True)
    print('Baseline performance')
    get_recall_rate(model)

    if adaptation_technique == 'simple':
        simple_adaptation(model, question_maxlen, answer_maxlen, qa)
    if adaptation_technique == 'kld':
        kld_adaptation(model, question_maxlen, answer_maxlen, qa, kld_weight)

