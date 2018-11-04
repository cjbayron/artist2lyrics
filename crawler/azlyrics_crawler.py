# Copyright 2018 Christopher John Bayron
# Copyright 2018 Pratyush Singh
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
# This file has been modified by Christopher John Bayron to support
# artist2lyrics operations. Original file is available in:
#
#     https://github.com/iiitv/lyrics-crawler

from random import shuffle
from re import findall, DOTALL
from crawler.base_crawler import CrawlerType1

class AZLyricsCrawler(CrawlerType1):
    def __init__(self, 
                 number_of_threads,
                 delayed_request=True,
                 max_errors=3):

        super().__init__('AZ Lyrics Crawler',
                         'http://azlyrics.com',
                         number_of_threads,
                         delay_request=delayed_request,
                         max_allowed_errors=max_errors)

    def get_albums_with_songs(self, raw_html):
        data = []

        album_html = findall(
            r'iv class=\"album\">(.*?)<d',
            raw_html,
            DOTALL
        )

        for content in album_html:
            album_name = findall(
                r'<b>\"(.*?)\"',
                content,
                DOTALL
            )

            if len(album_name) == 0:
                album_name = 'other'
            else:
                album_name = album_name[0]

            songs_with_url = findall(
                r'<a href=\"\.\.(.*?)\" target=\"_blank\">(.*?)</a><br>',
                content
            )
            data.append(
                (
                    album_name,
                    songs_with_url
                )
            )

        shuffle(data)
        return data

    def get_song_details(self, song_html):
        return findall(
            r'<div>.*?-->(.*?)</div>',
            song_html,
            DOTALL
        )[0].replace(
            '<br>',
            '\n'
        ).replace(
            '<i>',
            ''
        ).replace(
            '</i>',
            ''
        )
