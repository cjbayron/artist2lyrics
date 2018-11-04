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

from gensim.models import Word2Vec
import numpy as np

import os
import configparser
from argparse import ArgumentParser

from arch import rnn_keras
from common import constants as const
from common import utils as ut

if __name__ == "__main__":

    # Command line arguments
    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('-i', dest='input',
                            required=False,
                            help='Whitespace-separated input words for'
                                    ' lyrics generation')
    ARGS = ARG_PARSER.parse_args()

    input_words = []
    if ARGS.input is not None:
        init_lyrics = ARGS.input.strip()
        input_words = list(filter(None, init_lyrics.split(' ')))

    # read config
    config = configparser.ConfigParser()
    config.read(const.CONFIG_NAME)
    # number of stanzas to generate
    num_stanzas = int(config['generate']['num_stanzas'])
    min_words_in_stanza = int(config['generate']['min_words_in_stanza'])
    max_words_in_stanza = int(config['generate']['max_words_in_stanza'])

    # load word2vec model
    w2v_model_dir = config['embeddings']['model_dir']
    w2v_model_fn = config['embeddings']['model_fn']
    w2v_model_path = os.path.join(w2v_model_dir, w2v_model_fn)
    w2v_model = Word2Vec.load(w2v_model_path)
    # embedding list
    idx_to_emb = [ w2v_model[word] for word in w2v_model.wv.index2word ]
    # get index of period
    pd_idx = w2v_model.wv.vocab['.'].index

    # get indices to remove
    words_to_remove = ['<unk>']
    remove_idxs = [ w2v_model.wv.vocab[word].index for word in words_to_remove
                    if word in w2v_model.wv.vocab ]

    # initialize Logger
    log_dir = config['generate']['log_dir']
    fn_suffix = config['generate']['log_fn_suffix']
    logger = ut.Logger(log_dir, fn_suffix)

    # instantiate LSTM model
    lstm_rnn = rnn_keras.LyricsLSTM(config, logger, generate_mode=True)

    # perform prediction
    if not input_words:

        for i in range(num_stanzas):
            #cur_word_idx = pd_idx
            cur_word_idx = np.random.randint(low=0, high=len(w2v_model.wv.index2word))
            lyrics_str = w2v_model.wv.index2word[cur_word_idx]
            lstm_rnn.model.reset_states()
            # generate lyrics until period is encountered
            # as long as length is within min_words:max_words
            word_count = 0
            while( ( (cur_word_idx != pd_idx)
                      or (word_count < min_words_in_stanza) )
                    and (word_count < max_words_in_stanza) ):

                input_emb = idx_to_emb[cur_word_idx]
                # to prevent repeating
                remove_idxs_final = remove_idxs + [cur_word_idx]
                preds, probs = lstm_rnn.predict(input_emb, remove_idxs_final)
                probs = probs / np.sum(probs, axis=-1, keepdims=True)

                cur_word_idx = np.random.choice(preds,
                                                p=probs
                                                )
                lyrics_str = lyrics_str + " " + w2v_model.wv.index2word[cur_word_idx]
                word_count += 1

            logger.print(lyrics_str)

    else:
        # remove words not recognized by word2vec model
        filtered_words = [ word for word in input_words if word in w2v_model ]
        # convert words to embeddings
        input_embs = np.array([ w2v_model[word] for word in filtered_words ])        
        preds, probs = lstm_rnn.predict(input_embs, remove_idxs)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        cur_word_idx = np.random.choice(preds,
                                        p=probs
                                        )
        lyrics_str = ARGS.input.strip() + " " + w2v_model.wv.index2word[cur_word_idx]
        word_count = len(filtered_words) + 1

        while( ( (cur_word_idx != pd_idx)
                      or (word_count < min_words_in_stanza) )
                    and (word_count < max_words_in_stanza) ):

            input_emb = idx_to_emb[cur_word_idx]
            # to prevent repeating
            remove_idxs_final = remove_idxs + [cur_word_idx]
            preds, probs = lstm_rnn.predict(input_emb, remove_idxs_final)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            cur_word_idx = np.random.choice(preds,
                                            p=probs
                                            )
            lyrics_str = lyrics_str + " " + w2v_model.wv.index2word[cur_word_idx]
            word_count += 1

        logger.print(lyrics_str)