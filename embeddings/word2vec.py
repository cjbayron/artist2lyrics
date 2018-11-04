# MIT License
#
# Copyright (c) 2018 Christopher John Bayron
# Copyright (c) 2016 Kyubyong Park
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This file has been created by Christopher John Bayron based on "make_wordvectors.py"
# by Kyubyong Park. The referenced code is available in: 
#
#     https://github.com/Kyubyong/wordvectors

import numpy as np
from gensim.models import Word2Vec
import gensim.utils as gen_ut
import io
import os

min_word_freq = 1

def trim_rule(word, count, min_count):
    if min_count >= min_word_freq:
        return gen_ut.RULE_KEEP

def make_wordvectors(sentences, config):
    global min_word_freq

    vector_size = int(config['embeddings']['vector_size'])
    min_word_freq = int(config['embeddings']['min_word_freq'])
    num_negative = int(config['embeddings']['num_negative'])
    window_size = int(config['embeddings']['window_size'])
    num_epoch = int(config['embeddings']['num_epoch'])

    model_dir = config['embeddings']['model_dir']
    model_fn = config['embeddings']['model_fn']
    model_path = os.path.join(model_dir, model_fn)

    # this is for re-training pre-trained models on new text
    extend_model = bool(int(config['embeddings']['extend_model']))
    if extend_model is True:
        pre_trained_word2vec = config['embeddings']['pre_trained_model_path']
        model = Word2Vec.load(pre_trained_word2vec)
        model.build_vocab(sentences, update=True, trim_rule=trim_rule)
        model.train(sentences, total_examples=len(sentences), epochs=5)

    else:
        model = Word2Vec(sentences,
                         size=vector_size,
                         min_count=min_word_freq,
                         negative=num_negative, 
                         window=window_size,
                         iter=num_epoch)
    
    model.save(model_path)
    print("Saved Word2Vec model to %s!" % model_path)

    # word to embedding
    word_to_emb = { word:model[word] for idx, word in enumerate(model.wv.index2word) }

    return model, word_to_emb