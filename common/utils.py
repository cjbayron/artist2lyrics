# Copyright 2018 Christopher John Bayron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

import os
import math
from datetime import datetime

from . import constants as const

def time_batch_split(word_list, batch_size, bptt):
    """ Divides word_list into batches, then
    divides each batch according to BPTT
    """
    batched_list = np.array_split(word_list, batch_size)
    num_bptt = math.ceil(len(batched_list[0]) / bptt)

    bptt_batched_list = []
    for t in range(num_bptt):
        start = t * bptt
        end = start + bptt
        for batch in batched_list:
            if start == len(batch):
                bptt_batched_list.append([])
                continue

            if end > len(batch):
                bptt_batched_list.append(batch[start:])
            else:
                bptt_batched_list.append(batch[start:end])

    return bptt_batched_list

def word_list_to_emb_list(word_list, word_to_emb):
    """ Convert each word in word_list
    to its equivalent embedding vector
    """

    emb_list = []
    for word_seq in word_list:
        emb_seq = np.array([ word_to_emb[word] for word in word_seq ])
        # emb_seq = np.array([ word_to_emb[word] for word in word_seq 
        #                      if word in word_to_emb ])

        # need to flatten for saving as TFRecords
        emb_seq = emb_seq.flatten().tolist()
        emb_list.append(emb_seq)

    return emb_list

def word_list_to_idx_list(word_list, w2v_vocab):
    """ Convert each word in word_list
    to its equivalent index in w2v_vocab (Word2Vec model vocab)
    """

    idx_list = []
    for word_seq in word_list:
        # we add 1 to distinguish class 0 from padded zeros
        # (because we will pad zeros during training!)
        # idx_seq = [ (index2word.index(word) + 1) for word in word_seq ]
        idx_seq = [ (w2v_vocab.get(word).index + 1) for word in word_seq ]
        idx_list.append(idx_seq)

    return idx_list

def tsne_plot(model):
    # Copyright 2016 Jeff Delaney
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    #
    # Code from:
    #     https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
    #
    labels = []
    tokens = []

    print("Displaying t-SNE...")

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40,
                      n_components=2,
                      init='pca',
                      n_iter=2500,
                      random_state=23)

    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.title("t-SNE")
    plt.show()

def convert_to_feature(value, feature_type, is_list=False):
    """ Convert input value to TF Feature based on given Feature type
    """

    feature = None
    if not is_list:
    	value = [value]

    if feature_type == 'int64':
        feature = tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    elif feature_type == 'bytes':
        feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    elif feature_type == 'float':
        feature = tf.train.Feature(float_list=tf.train.FloatList(value=value))

    else:
        raise Exception("Feature not supported!")

    return feature

def save_seq2seq_as_TFRecord(path, in_seq_list, out_seq_list):
    """ Save input and output sequences as TensorFlow records
    """

    writer = tf.python_io.TFRecordWriter(path)
    for in_seq, out_seq in zip(in_seq_list, out_seq_list):

        if len(in_seq) % len(out_seq) != 0:
            raise Exception("Input and Output sequence length mismatch!")

        feature_map = {
            'input_words': convert_to_feature(in_seq, 'float', is_list=True),
            'next_words': convert_to_feature(out_seq, 'int64', is_list=True),
            'seq_len': convert_to_feature(len(out_seq), 'int64')
        }

        seq_ex = tf.train.Example(features=tf.train.Features(feature=feature_map))
        # Serialize Example, then write as TFRecord
        writer.write(seq_ex.SerializeToString())

    writer.close()

def example_parser(serialized_example):
    """ Parses an Example in a TFRecord
    """

    # define Features of Example to parse
    feature_set = {
        # use FixedLenSequenceFeature for array
        'input_words': tf.FixedLenSequenceFeature([], dtype=tf.float32,
                                                allow_missing=True),

        'next_words': tf.FixedLenSequenceFeature([], dtype=tf.int64,
                                                allow_missing=True),

        'seq_len' : tf.FixedLenFeature([], dtype=tf.int64),
    }

    # parse Example
    features = tf.parse_single_example(serialized_example, features=feature_set)

    input_words = features['input_words']
    next_words = features['next_words']
    seq_len = features['seq_len']

    return input_words, next_words, seq_len

def tfg_read_from_TFRecord(records_file, buf_size, batch_size, shuffle):
    """ Converts data from TFRecords file into a Dataset,
    and creates an Iterator for consuming a batch of the Dataset
    """

    # create dataset from TFRecords
    dataset = tf.data.TFRecordDataset(records_file)
    # parse record into tensors
    dataset = dataset.map(example_parser)
    # shuffle dataset
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=buf_size)
    # repeat occurence of inputs
    dataset = dataset.repeat()
    # generate batches
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes = ([None],
                                                    [None],
                                                    []))
    # create one-shot iterator
    data_it = dataset.make_one_shot_iterator()
    # get features
    input_words, next_words, seq_len = data_it.get_next()
    return input_words, next_words, seq_len

def save_tf_model(step, alias, tf_saver, sess):
    """ Saves current values of trained model
    """

    checkpoint_path = "models/" + const.TF_MODELS_PREFIX

    # filename format arrangement: date, alias, epoch
    checkpoint_file = checkpoint_path.format(
        datetime.today().strftime(const.DATETIME_FORMAT),
        alias,
        step
    )

    save_path = tf_saver.save(sess, checkpoint_file)
    print("Tensorflow graph saved in path: %s" % save_path)

    return save_path

def get_keras_model_save_path(step, alias):
    """ Pre-formats filename for Keras model
    """

    save_path = "models/" + const.K_MODELS_PREFIX

    # filename format arrangement: date, alias, epoch
    save_path = save_path.format(
        datetime.today().strftime(const.DATETIME_FORMAT),
        alias,
        step
    )

    return save_path

class Logger():
    """ Logging utility class
    """

    def __init__(self, log_dir, fn_suffix):
        self.log_file = None
        log_path = os.path.join(log_dir,
                                "{0}_{1}".format(
                                    datetime.today().strftime(const.DATETIME_FORMAT),
                                    fn_suffix
                                    )
                                )

        self.log_file = open(log_path, 'w')

    def __del__(self):
        if self.log_file is not None:
            self.log_file.close()

    def print(self, str_to_print):
        """ Add datetime then print to file and terminal
        """
        dt_now = '[' + datetime.today().strftime(const.LOG_DATETIME_FORMAT)[:-3] + '] '
        str_to_print = dt_now + str_to_print
        self.log_file.write(str_to_print + '\n')
        print(str_to_print)