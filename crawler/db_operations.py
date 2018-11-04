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

from html import unescape
from psycopg2 import connect

def get_connection():
    # use default DB 'postgres'
    conn = connect(database='postgres')
    return conn, conn.cursor()


def create():
    sql = '''CREATE TABLE IF NOT EXISTS songs (
              id BIGSERIAL PRIMARY KEY NOT NULL ,
              song TEXT,
              song_url VARCHAR(512),
              album TEXT,
              album_url VARCHAR(512),
              start_url VARCHAR(512),
              lyrics TEXT,
              singers TEXT,
              director TEXT,
              lyricist TEXT,
              last_crawled TIMESTAMP,
              last_updated TIMESTAMP
            );'''

    conn, cur = get_connection()
    cur.execute(sql)
    conn.commit()
    conn.close()


def save(song, song_url, album, album_url, start_url, lyrics, singers,
         director, lyricist):
    song, album, lyrics, singers, director, lyricist = unescape(song), \
                                                       unescape(album), \
                                                       unescape(lyrics), \
                                                       unescape(str(singers)), \
                                                       unescape(str(director)), \
                                                       unescape(str(lyricist))

    sql = """SELECT id FROM songs WHERE song_url=%s AND start_url=%s;"""

    conn, cur = get_connection()

    cur.execute(
        sql,
        (
            song_url,
            start_url
        )
    )

    result = cur.fetchall()
    if len(result) == 0:
        sql = """INSERT INTO songs(
                    song, song_url, album, album_url, start_url, lyrics,
                    singers, director, lyricist, last_updated, last_crawled
                  )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP,
                 CURRENT_TIMESTAMP) RETURNING id;"""

        cur.execute(
            sql,
            (
                song,
                song_url,
                album,
                album_url,
                start_url,
                lyrics,
                str(singers),
                str(director),
                str(lyricist)
            )
        )
    else:
        sql = """UPDATE songs SET song=%s, song_url=%s, album=%s,
        album_url=%s, start_url=%s, lyrics=%s, singers=%s, director=%s,
        lyricist=%s, last_updated=CURRENT_TIMESTAMP,
        last_crawled=CURRENT_TIMESTAMP WHERE id=%s RETURNING id;"""

        cur.execute(
            sql,
            (
                song,
                song_url,
                album,
                album_url,
                start_url,
                lyrics,
                str(singers),
                director,
                lyricist,
                result[0][0]
            )
        )

    result = cur.fetchall()[0][0]
    conn.commit()
    conn.close()
    return result


def fetch_all_lyrics():
    """
    Get all lyrics from database
    """
    sql = """SELECT lyrics FROM songs;"""
    conn, cur = get_connection()

    cur.execute(sql)
    result = cur.fetchall()

    conn.close()

    return result

def fetch_all_lyrics_from_artists(artists):
    """
    Get all lyrics of particular artists
    """
    artists_str = ','.join(["\'" + artist + "\'" for artist in artists])
    
    sql = """SELECT lyrics FROM songs WHERE singers IN (%s);""" % artists_str
    #sql = """SELECT lyrics FROM songs WHERE is_tagalog AND singers IN (%s);""" % artists_str
    conn, cur = get_connection()
    
    cur.execute(sql)
    result = cur.fetchall()

    conn.close()

    return result

def load(id):
    sql = """SELECT * FROM songs WHERE id=%s;"""

    conn, cur = get_connection()

    cur.execute(
        sql,
        (
            id,
        )
    )
    result = cur.fetchall()

    conn.close()

    return result[0][1:]


def number_of_songs(start_url, url):
    sql = """SELECT count(*) FROM songs WHERE start_url=%s AND album_url=%s;"""

    conn, cur = get_connection()

    cur.execute(
        sql,
        (
            start_url,
            url
        )
    )

    result = cur.fetchall()[0][0]
    conn.close()
    return result


def exists_song(start_url, url):
    conn, cur = get_connection()

    cur.execute(
        'SELECT * FROM songs WHERE start_url=%s AND song_url=%s;',
        (
            start_url,
            url
        )
    )

    result = cur.fetchall()

    conn.close()

    return len(result) > 0
