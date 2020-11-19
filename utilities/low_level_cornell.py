import os
"""
Logging provides a set of convenience functions for simple logging usage.
These are debug(), info(), warning(), error() and critical().
"""
import logging
import utils
"""
getLogger() returns a reference to a logger instance with the specified name if it is provided, or root if not
"""
log = logging.getLogger("cornell")
DATA_DIR = "./cornell"
SEPARATOR = "+++$+++"

"""
Using this function we open the file from the directory.
We convert each line to string and the 'yield' function is a bit tricky.
It is used in body of a generator function and returns a generator(Iterable).
Each line is split by the specified separator and str.strip removes the leading or
trailing spaces.This is done for each line and the whole thing is converted into
an iterable list and returned using yield.
"""
def iterate_entries(data_dir,text_file):
    with open(os.path.join(data_dir,text_file),"rb") as f:
        for each in f:
            each = str(each,encoding="utf-8",errors="ignore")
            yield list(map(str.strip,each.split(SEPARATOR)))

"""
This function gives out the set of movies which has the genre that we
requested.
"""
def get_filtered_set(data_dir,genre_filter):
    movies = set()
    for each in iterate_entries(data_dir,"movie_titles_metadata.txt"):
        m_id = each[0]
        m_genres = each[5]
        #'.find()' finds first occurence of specified value.
        #Returns -1 if not found.
        if m_genres.find(genre_filter) != -1:
            movies.add(m_id)
    return movies

"""
movies_line.txt has line_id,movie_id,speaker_name and movie_dialogue.
We just access the movie dialogue and tokenize it.
Our output is a dictionary for which keys are dialogue_id and values are phrase tokens.
"""
def read_phrases(data_dir,movies=None):
    out = {}
    for each in iterate_entries(data_dir,"movie_lines.txt"):
        dialogue_id = each[0]
        m_id = each[2]
        dialogue = each[4]
        #If it doesn't find the filtered genre and movie id,
        #it goes to the top level loop ignoring next lines.
        if movies and m_id not in movies:
            continue
        tokens = utils.tokenize(dialogue)
        if tokens:
            out[dialogue_id] = tokens
    return out

"""
movie_conversations.txt has a list of the line_ids which make a conversation together.
"""

def load_conversations(data_dir,lines,movies=None):
    out = []
    for each in iterate_entries(data_dir,"movie_conversations.txt"):
        m_id = each[2]
        convo = each[3]
        if movies and m_id not in movies:
            continue
        #Take them out of list.
        dialogue_ids = convo.strip("[]").split(", ")
        #Strip them off the quotations and make them into list of individual words.
        dialogue_ids = list(map(lambda s: s.strip("'"),dialogue_ids))
        #Getting all these lines together in a list.
        dialogue = [lines[dialogue_id] for dialogue_id in dialogue_ids if dialogue_id in lines]
        if dialogue:
            out.append(dialogue)
    return out




"""
Data_dir can also be given as an argument, if different.
genre_filter is optional as well.
load_dialogues loads dialogues from cornell dataset.
Returns list of list of list of words.

"""
def load_dialogues(data_dir=DATA_DIR, genre_filter=''):
    filter_movie = None
    if genre_filter:
        filter_movie = get_filtered_set(data_dir,genre_filter)
        log.info("Loaded %d movies belonging to %s genre",len(filter_movie),genre_filter)
    log.info("Reading and tokenizing phrases...")
    lines = read_phrases(data_dir, movies=filter_movie)
    log.info("Loaded %d phrases",len(lines))
    dialogues = load_conversations(data_dir,lines,filter_movie)
    return dialogues

"""
Takes data directory as input and gives out a dictionary of
movie ids mapped to their respective genres.
"""
def read_genres(data_dir):
    out = {}
    for each in iterate_entries(data_dir, "movie_titles_metadata.txt"):
        m_id, m_genres = each[0], each[5]
        l_genres = m_genres.strip("[]").split(", ")
        l_genres = list(map(lambda s: s.strip("'"), l_genres))
        out[m_id] = l_genres
    return out


#print(read_genres('./cornell'))
#print(load_dialogues('./cornell','comedy'))
