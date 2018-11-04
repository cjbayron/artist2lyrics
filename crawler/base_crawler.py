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

import os
from queue import LifoQueue
from threading import Thread

import crawler.db_operations as db_operations
import crawler.print_util as print_util
from crawler.network_manager import open_request
from crawler.print_util import Colors

class BaseCrawler:
    def __init__(self, name, start_url, number_of_threads, max_err=10,
                 delay_request=False):
        """
        Base class for all other crawlers. This class contains all information
        that will be common in all crawlers.
        :param name: Name of crawler
        :param start_url: Base URL of website
        :param number_of_threads: Number of threads to use to crawl
        :param max_err: Max number of allowed errors for a crawl
        :param delay_request: Whether to delay while making requests or not
        """
        self.delay_request = delay_request
        self.name = name
        self.start_url = start_url
        self.number_of_threads = number_of_threads
        self.max_allowed_errors = max_err

class CrawlerType1(BaseCrawler):
    def __init__(self, name, start_url, number_of_threads,
                 delay_request=False, max_allowed_errors=3):
        """

        :param name: As usual
        :param start_url: As usual
        :param number_of_threads: As usual
        :param delay_request: As usual
        :param max_allowed_errors: As usual
        """
        super().__init__(name, start_url, number_of_threads=number_of_threads,
                         delay_request=delay_request,
                         max_err=max_allowed_errors)
        self.task_queue = LifoQueue()

        # create database
        db_operations.create()

    def run(self, artists_list):
        """
        Method called from subclasses to start crawling process
        """
        # Crawl cycle starts
        print_util.print_info(
            'Starting new crawl with {0}'.format(
                self.name
            ),
            Colors.BLACK
        )

        url_artist_pairs = [(os.path.join(artist_name[0], artist_name + '.html'),
                             artist_name) for artist_name in artists_list]

        for url, artist in url_artist_pairs:
            self.task_queue.put(
                {
                    'type': 1,
                    'url': url,
                    'artist': artist,
                    'n_errors': 0
                }
            )

        # Start all threads
        threads = []
        for n in range(1, self.number_of_threads + 1):
            temp_thread = Thread(
                target=self.threader,
                args=(n,)
            )
            threads.append(temp_thread)
            temp_thread.start()
        for temp_thread in threads:
            temp_thread.join()
            # Crawl cycle ends

    def threader(self, thread_id):
        """
        Worker function
        :param thread_id: As usual
        """
        while not self.task_queue.empty():
            task = self.task_queue.get()

            if task['n_errors'] >= self.max_allowed_errors:
                print_util.print_warning(
                    '{0} --> Too many errors in task {1}. Skipping.'.format(
                        thread_id,
                        task
                    )
                )
                continue

            print_util.print_info(
                '{0} --> New task : {1}'.format(
                    thread_id,
                    task
                )
            )

            try:
                if task['type'] == 1:
                    self.get_artist_albums(
                        thread_id,
                        task['url'],
                        task['artist']
                    )
                elif task['type'] == 2:
                    self.get_song(
                        thread_id,
                        task['url'],
                        task['song'],
                        task['album'],
                        task['album_url'],
                        task['artist']
                    )

                print_util.print_info(
                    '{0} --> Task complete : {1}'.format(
                        thread_id,
                        task
                    ),
                    Colors.GREEN
                )
            except Exception as e:
                print_util.print_error(
                    '{0} --> Error : {1}'.format(
                        thread_id,
                        e
                    )
                )
                task['n_errors'] += 1
                self.task_queue.put(task)

    def get_artist_albums(self, thread_id, url, artist):
        """
        Method to get all songs for an artist
        :param thread_id: As usual
        :param url: As usual
        :param artist: Artist name
        """
        website = self.start_url + '/' + url
        raw_html = open_request(website, delayed=self.delay_request)

        albums_with_songs = self.get_albums_with_songs(raw_html)

        for album, song_with_url in albums_with_songs:
            for song_url, song in song_with_url:
                self.task_queue.put(
                    {
                        'type': 2,
                        'song': song,
                        'url': song_url,
                        'album': album,
                        'album_url': url,
                        'artist': artist,
                        'n_errors': 0
                    }
                )

    def get_song(self, thread_id, url, song, album, album_url, artist):
        """
        Method to get details of a song and save in database
        :param thread_id: As usual
        :param url: As usual
        :param song: Song title
        :param album: Album name
        :param album_url: URL of album (same as artist) on the website
        :param artist: As usual
        """
        if db_operations.exists_song(self.start_url, url):
            print_util.print_warning(
                '{0} -> Song {1} already exists. Skipping'.format(
                    thread_id,
                    song
                )
            )
            return

        print_util.print_info("get_song: 1\n")
        song_website = self.start_url + url
        song_html = open_request(song_website, delayed=self.delay_request)
        print_util.print_info("get_song: 2\n")
        lyrics = self.get_song_details(song_html)
        print_util.print_info("get_song: 3\n")
        db_operations.save(
            song=song,
            song_url=url,
            album=album,
            album_url=album_url,
            start_url=self.start_url,
            lyrics=lyrics,
            singers=artist,
            director=artist,
            lyricist=artist
        )

    def get_albums_with_songs(self, raw_html):
        """
        Get all songs with albums for an artist
        :param raw_html: Web page HTML code
        :return: Songs with URL and album
        """
        return [
            (
                'album1',
                [
                    ('url1', 'song1'),
                    ('url2', 'song2')
                ]
            ),
            (
                'album2',
                [
                    ('url3', 'song3'),
                    ('url4', 'song4')
                ]
            )
        ]

    def get_song_details(self, song_html):
        """
        Get lyrics of the song from webpage
        :param song_html:
        :return:
        """
        return 'la la la la'