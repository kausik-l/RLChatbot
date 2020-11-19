"""
Collections module in python has many containers that we can use like counters, ordered dictionary, chain map, etc.
"""
import collections
import os,sys,logging
"""
Itertools module works as a fast, memory-efficient tool that is used in combination to form iterator algebra.
"""
import itertools
import pickle

from . import low_level_cornell

"""
We'll replace all words which occur less than 10 times with #UNK to save some memory and time.
Training pairs will be created with 20 tokens to reduce no. of operations and memory.
"""
UNKNOWN_TOKEN = '#UNK'
BEGIN_TOKEN = '#BEG'
END_TOKEN = '#END'
MAX_TOKENS = 20
MIN_TOKEN_FREQ = 10

EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"

log = logging.getLogger("data")

"""
We save the file which maps tokens to some integer IDs.
(word -> ID)
"""
def save_emb_dict(dir_name,emb_dict):
    with open(os.path.join(dir_name, EMB_DICT_NAME),"wb") as f:
        pickle.dump(emb_dict,f)

"""
To load the file which has tokens mapped to some integer IDs.
"""
def load_emb_dict(dir_name):
    with open(os.path.join(dir_name,EMB_DICT_NAME),"rb") as f:
        return pickle.load(f)

"""
List of words and embeddings dictionary are given as inputs.
It returns list of integer IDs.
"""
def encode_words(words,emb_dict):
    #Starting token.
    out = [emb_dict[BEGIN_TOKEN]]
    unknown_idx = emb_dict[UNKNOWN_TOKEN]
    for each in words:
        #get() method gets the value of specified key. If key does not exist, it will return the value(2nd argument).
        idx = emb_dict.get(each.lower(),unknown_idx)
        out.append(idx)
    #Ending token.
    out.append(emb_dict[END_TOKEN])
    return out

"""
Converts list of phrase pairs(list of (phrase,phrase)) to
list of tuples ([input_phrase_id_sequence],[output_phrase_id_sequence])
"""
def encode_phrase_pairs(phrase_pairs, emb_dict, filter_unknowns=True):
    unknown_tkn = emb_dict[UNKNOWN_TOKEN]
    out = []
    for p1,p2 in phrase_pairs:
        #Tuple of 2 lists
        p = encode_words(p1,emb_dict), encode_words(p2,emb_dict)
        #If we encounter unknown tokens, go to the start of loop again.
        if unknown_tkn in p[0] or unknown_tkn in p[1]:
            continue
        out.append(p)
    return out

"""
We group the training data(list of (seq1,seq2) pairs) by first phrase.
It returns a list of (seq1,[seq2,seq3,...]) pairs.
defaultdict overrides one method and adds one writable instance variable.
The output we get is similar to this:
if s = [('hello','hru'),('bye','goodbye'),('hello','hi there!')]
it will be converted to [('hello',['hru','hi there!']),('bye','goodbye')]
grouped by the 1st phrase.
"""
def group_train_data(training_data):
    groups = collections.defaultdict(list)
    for p1,p2 in training_data:
        l = groups[tuple(p1)]
        l.append(p2)
    return list(groups.items())

"""
Iterates batches of given size.
Input : Data and the batch size.
Output: A generator variable with batches.
"""
def iterate_batches(data,batch_size):
    #assert isinstance() can be used to check whether the object belongs to certain class.
    #If this is false, it will lead to assertation error.
    assert isinstance(data,list)
    assert isinstance(batch_size,int)

    count = 0
    while True:
        batch = data[count*batch_size:(count+1)*batch_size]
        if len(batch)<=1:
            break
        #generator object will be returned(We use this instead of appending to another variable).
        yield batch
        count += 1

"""
Loads dialogues and converts it into phase-reply pairs suitable for training.
Inputs : Genre filter(Optional), max no. of tokens , the minimum token frequency.
We'll replace all words which occur less than 10 times with #UNK to save some memory and time.
Training pairs will be created with 20 tokens to reduce no. of operations and memory.
Outputs : list of (phrase,phrase) pairs and dictionary with each word as key and a unique ID as value.
"""
def load_data(genre_filter,max_tokens=MAX_TOKENS,min_token_frequency=MIN_TOKEN_FREQ):
    dialogues = low_level_cornell.load_dialogues(genre_filter=genre_filter)
    if not dialogues:
        log.error("No dialogues found!")
        sys.exit()
    log.info("Loaded %d dialogues with %d phrases, creating training pairs",len(dialogues),sum(map(len,dialogues)))
    phrase_pairs = dialogues_to_pairs(dialogues,max_tokens=max_tokens)
    log.info("Counting frequency of the words....")
    #Counts the frequency of each word.(For example,{'red':4,'how':5,'not':10}. The output will be like this.)
    word_counts = collections.Counter()
    for each in dialogues:
        for phrase in each:
            #By updating, you are sending that particular phrase(List of word tokens) to the counter.
            word_counts.update(phrase)
    """
    map(function, iterable). Takes each element of iterable and calls the function with it.
    filter(function,sequence). Function checks if each element of sequence is True or False.
    Sequence is the sequence we want to filter.
    word_count.items() converts that dictionary into list of (element,count) pairs.
    phrase[1] refers to the freq. number and phrase[0] is the actual word.
    So freq_set will be a set of words which are repeated more than 10 times(min_token_frequency).
    """
    freq_set = set(map(lambda phrase:phrase[0], filter(lambda phrase:phrase[1] >= min_token_frequency, word_counts.items())))
    log.info("Data has %d unique words, %d of them occur more than %d times", len(word_counts),len(freq_set), min_token_frequency)
    #Dictionary of words with unique ID.
    phrase_dict = phrase_pair_dict(phrase_pairs, freq_set)
    return phrase_pairs,phrase_dict

"""
Inputs: list of (phrase,phrase) pairs and the set of words which are repeated more than 10 times.
Output : dictionary with words as keys and a unique ID as value.
"""

def phrase_pair_dict(phrase_pairs,freq_set):
    out = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
    #next_id assigns an ID to the next word other than the above 3.
    next_id = 3
    for p1,p2 in phrase_pairs:
        #itertools.chain() splits the elements from the 2 lists.
        #Ex: chain('ABC','DEF') --> A B C D E F.
        for each in map(str.lower,itertools.chain(p1,p2)):
            #If it's a word from freq_set, give it an ID.
            if each not in out and each in freq_set:
                out[each] = next_id
                next_id += 1
    return out

"""
Inputs: Dialogues from low_level_cornell.py load_dialogues() function.(List of list of list of word tokens) and in our case max tokens is 20.
Output: The output has list of (phrase,phrase) pairs which are less than max_tokens(20 in our case). This makes the replies shorter than 20 words.
"""
def dialogues_to_pairs(dialogues,max_tokens=None):
    out = []
    for dialogue in dialogues:
        prev_phrase = None
        for phrase in dialogue:
            if prev_phrase is not None:
                if max_tokens is None or (len(prev_phrase) <= max_tokens and len(phrase) <= max_tokens):
                    out.append((prev_phrase,phrase))
            prev_phrase = phrase
    return out


def decode_words(indices,rev_emb_dict):
    #.get() returns value of the item with specified key
    return [rev_emb_dict.get(idx,UNKNOWN_TOKEN) for idx in indices]

"""
Inputs: word tokens and the end token.
Outputs: list of tokens.
"""

def trim_tokens_seq(tokens,end_token):
    out = []
    for each in tokens:
        out.append(each)
        #If #END token is encountered it indicated the end of phrase.So we break after the end.
        if each == end_token:
            break
    return out

"""
We split the whole data into training and test set.
Inputs: data and split ratio.
Output : Train and Test data.
"""
def split_train_test(data,train_ratio=0.95):
    part = int(len(data) * train_ratio)
    return data[:part],data[part:]
