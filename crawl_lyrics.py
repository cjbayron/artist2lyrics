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

from crawler import azlyrics_crawler
from common import constants as const

if __name__ == '__main__':

    # read config
    config = configparser.ConfigParser()
    config.read(const.CONFIG_NAME)
    # comma-separated string of artists
    artists_csv = config['crawler']['artists']
    artists_list = list(filter(None, artists_csv.split(',')))

    # initialize crawler class
    crawler = azlyrics_crawler.AZLyricsCrawler(number_of_threads=1,
                                               max_errors=5,
                                               delayed_request=True)
    # start crawling songs of artists
    crawler.run(artists_list)