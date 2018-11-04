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

import keras.models as K_model
import keras.layers as K_layer
import keras.utils as K_ut
import keras.callbacks as K_callback
import keras.backend as K_bak
from tensorflow import reshape as tf_reshape
from tensorflow.python.training.saver import Saver as tf_Saver
from tensorflow.python.ops.nn_ops import top_k as tf_top_k
import numpy as np

import os
import math

from common import utils as ut
from common import constants as const

class MyCallback(K_callback.Callback):
    """ Callback function for Keras model.fit()
    """

    def __init__(self, logger):
        self.logger = logger
        super(MyCallback, self).__init__()
    
    def on_batch_end(self, batch, logs={}):
        global real_epoch_count, num_epoch, batch_num
        self.logger.print("Epoch: %d/%d, Batch: %d/%d, Loss: %f"
                          % (real_epoch_count+1, num_epoch,
                             batch+1, batch_num, logs['loss']))
        return

    def on_epoch_end(self, epoch, logs={}):
        global real_epoch_count
        real_epoch_count += 1
        return

class LyricsLSTM():
    """ Many-to-many RNN architecture for lyrics generation
    """

    def __init__(self, config, logger, generate_mode=False):
        # set config
        self.config = config
        self.generate_mode = generate_mode
        # hyperparameters
        self.embedding_size = int(config['embeddings']['vector_size'])
        self.vocab_size = int(config['embeddings']['vocab_size'])
        self.lstm_units_1 = int(config['LyricsLSTM']['lstm_units_1'])
        self.lstm_units_2 = int(config['LyricsLSTM']['lstm_units_2'])
        self.dropout_1 = float(config['LyricsLSTM']['lstm_dropout_1'])
        self.dropout_2 = float(config['LyricsLSTM']['lstm_dropout_2'])
        # logger
        self.logger = logger

        self.model = None
        if (generate_mode):
            self.model = self.__init_prediction_graph()
        else:
            self.model = self.__init_training_graph()

    def __init_training_graph(self):
        config = self.config
        self.num_epoch = int(config['training']['num_epoch'])
        num_seq = int(config['embeddings']['num_seq'])
        trn_batch_size = int(config['training']['batch_size'])
        retrain_model = bool(int(config['training']['retrain_model']))

        # set steps for training
        global batch_num
        self.steps_per_epoch = math.ceil(num_seq / trn_batch_size)
        batch_num = self.steps_per_epoch
        # initialize input source (TF dataset generator)
        tf_rec_path = os.path.join(config['training']['tf_rec_dir'],
                                   config['training']['tf_rec_fn'])
        emb_seq, next_w_seq, seq_len = \
            ut.tfg_read_from_TFRecord(tf_rec_path,
                                      buf_size=num_seq,
                                      batch_size=trn_batch_size,
                                      shuffle=False)
        # reshape tensors
        tf_rec_input = tf_reshape(emb_seq,
                                  [trn_batch_size, -1, self.embedding_size])
        tf_labels = tf_reshape(next_w_seq,
                               [trn_batch_size, -1])
        # initialize layers
        model_input = K_layer.Input(tensor=tf_rec_input,
                                    batch_shape=(trn_batch_size,
                                                 None,
                                                 self.embedding_size))
        lstm_layer_1 = K_layer.LSTM(units=self.lstm_units_1,
                                    dropout=self.dropout_1,
                                    stateful=True,
                                    return_sequences=True)
        lstm_layer_2 = K_layer.LSTM(units=self.lstm_units_2,
                                    dropout=self.dropout_2,
                                    stateful=True,
                                    return_sequences=True)
        dense_layer = K_layer.Dense(units=self.vocab_size + 1,
                                    activation='softmax')
        # graph flow
        lstm_1_out = lstm_layer_1(model_input)                                  
        lstm_2_out = lstm_layer_2(lstm_1_out)
        softmax_out = K_layer.TimeDistributed(dense_layer)(lstm_2_out)

        # initialize Keras model
        model = K_model.Model(inputs=model_input, outputs=softmax_out)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      target_tensors=[tf_labels])

        if retrain_model:
            # restore weights via TF Saver
            checkpoint_path = config['training']['retrain_model_path']
            self.logger.print("Re-training model in %s..." % checkpoint_path)
            saver = tf_Saver()
            sess = K_bak.get_session()
            saver.restore(sess, checkpoint_path)

        return model

    def __init_prediction_graph(self):
        config = self.config
        # load Keras model and restore weights via TF Saver
        checkpoint_path = config['generate']['weights_path']
        self.logger.print("Loading weights in %s..." % checkpoint_path)
        keras_model_path = config['generate']['model_path']
        trained_model = K_model.load_model(keras_model_path)
        saver = tf_Saver()
        sess = K_bak.get_session()
        saver.restore(sess, checkpoint_path)

        # get weights of trained model
        lstm_1_w = trained_model.get_layer('lstm_1').get_weights()
        lstm_2_w = trained_model.get_layer('lstm_2').get_weights()
        td_w = trained_model.get_layer('time_distributed_1').get_weights()
        dense_units = td_w[0].shape[1]

        # NOW create model with diff batch size
        # then transfer weights of trained model
        model_input = K_layer.Input(batch_shape=(1,
                                                 None,
                                                 self.embedding_size))
        lstm_layer_1 = K_layer.LSTM(units=self.lstm_units_1,
                                    stateful=True,
                                    return_sequences=True)
        lstm_layer_2 = K_layer.LSTM(units=self.lstm_units_2,
                                    stateful=True,
                                    return_sequences=True)
        dense_layer = K_layer.Dense(units=dense_units,
                                    activation='softmax')
        # graph flow
        lstm_1_out = lstm_layer_1(model_input)                                  
        lstm_2_out = lstm_layer_2(lstm_1_out)
        softmax_out = K_layer.TimeDistributed(dense_layer)(lstm_2_out)                                                        
        # initialize Keras model
        model = K_model.Model(inputs=model_input, outputs=softmax_out)
        # transfer weights
        model.get_layer('lstm_1').set_weights(lstm_1_w)
        model.get_layer('lstm_2').set_weights(lstm_2_w)
        model.get_layer('time_distributed_1').set_weights(td_w)

        return model

    def __save_models(self):
        config = self.config
        global real_epoch_count

        # use TensorFlow Saver to properly save Keras model
        # -> When using only Keras save(), different behavior
        #    is encountered in loaded model (via load_model())
        saver = tf_Saver()
        sess = K_bak.get_session()
        tf_save_path = ut.save_tf_model(real_epoch_count,
                                        self.__class__.__name__,
                                        saver, sess)
        keras_save_path = ut.get_keras_model_save_path(real_epoch_count,
                                                       self.__class__.__name__)
        self.model.save(keras_save_path)
        self.logger.print("Keras model saved in path: %s" % keras_save_path)

        # update config
        config['generate']['weights_path'] = tf_save_path
        config['generate']['model_path'] = keras_save_path
        config['training']['retrain_model_path'] = tf_save_path
        with open(const.CONFIG_NAME, 'w') as config_file:
            config.write(config_file)

    def train(self):
        if self.generate_mode:
            raise Exception("Cannot perform training in generate mode!")

        config = self.config
        save_epoch_interval = int(config['training']['save_epoch_interval'])
        save_many = bool(int(config['training']['save_many']))

        callback = MyCallback(self.logger)
        global real_epoch_count, num_epoch
        real_epoch_count = 0
        num_epoch = self.num_epoch

        self.logger.print("Training %s on %d batches for %d epochs..."
                          % (self.__class__.__name__, self.steps_per_epoch, num_epoch))

        # we perform loop so we can reset the state after each epoch
        for e in range(num_epoch):
            self.model.fit(epochs=1,
                           steps_per_epoch=self.steps_per_epoch,
                           callbacks=[callback],
                           verbose=0)
            # reset state after every epoch
            self.model.reset_states()

            if (save_many is True) and \
               ( ((e+1) % save_epoch_interval == 0) and ((e+1) != num_epoch) ):
                self.__save_models()
                
        self.__save_models()

    def predict(self, inputs, remove_idxs=[]):
        if not self.generate_mode:
            raise Exception("Can only perform prediction in generate mode!")

        config = self.config
        sampling_size = int(config['generate']['sampling_size'])
        temp = float(config['generate']['temperature'])
        # reshape inputs, then get prediction
        nd_inputs = np.reshape(inputs, (1, -1, self.embedding_size))
        softmax_out = self.model.predict(nd_inputs, verbose=0)

        # apply temperature
        temped_exps = (softmax_out)**(1/temp)
        temped_sm = temped_exps / np.sum(temped_exps, axis=-1, keepdims=True)

        # remove zero padding
        sm_nopad = temped_sm[0, :, 1:]
        # zero-out unwanted indices (e.g. <unk>)
        for idx in remove_idxs:
            sm_nopad[:, idx] = 0
        # get only last output
        probs = sm_nopad[-1]
        # get only words with highest probability
        if sampling_size > len(probs):
            sampling_size = len(probs)
        preds = tf_top_k(probs, k=sampling_size)
        sess = K_bak.get_session()
        probs, preds = sess.run(preds)

        return preds, probs