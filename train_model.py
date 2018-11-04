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

import configparser

from arch import rnn_keras
from common import constants as const
from common import utils as ut

if __name__ == "__main__":

    # read config
    config = configparser.ConfigParser()
    config.read(const.CONFIG_NAME)
    # initialize Logger
    log_dir = config['training']['log_dir']
    fn_suffix = config['training']['log_fn_suffix']
    logger = ut.Logger(log_dir, fn_suffix)

    # instantiate model
    lstm_rnn = rnn_keras.LyricsLSTM(config, logger)

    # perform training
    lstm_rnn.train()