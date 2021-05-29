from nltk.tokenize import TweetTokenizer
from nltk.translate import bleu_score
import string

"""
We give a sentence and it tokenizes it.
When we set preserve_case to False it will downcase the string.
It is True by default.
"""
def tokenize(s):
    return TweetTokenizer(preserve_case=False).tokenize(s)

"""
Input : candidate sequence should be provided as a list of tokens and
        reference sequences in the form of list of references where each
        reference is list of tokens again.
Output : sentence_bleu calculates the score.

BLEU score will vary from 0 to 1.

SmoothingFunction is used when sentences are smaller.
Smoothing method 2: Adds 1 to both numerator and denominator from Chin-Yew Lin and Franz Josef Och (2004)
Automatic evaluation of machine translation quality using longest common subsequence
and skip-bigram statistics. In ACL04(http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf).

"""
def calc_bleu_many(cand_sequence,ref_sequences):
    return bleu_score.sentence_bleu(ref_sequences,cand_sequence,
                                    smoothing_function=bleu_score.SmoothingFunction().method1,
                                    weights=(0.5,0.5))

"""
When we have just one generated sequence and one reference sequence.
We calculate bleu score with the other function we created here.

Input : candidate sequence should be provided as a list of tokens and
        reference sequences in the form of list of tokens.
Output : sentence_bleu calculates the score.

"""
def calc_bleu(cand_sequence,ref_sequence):
    return calc_bleu_many(cand_sequence,[ref_sequence])

"""
Input : List of Tokens.
Output : Corresponding String.
"""
def untokenize(tokens):
    return "".join([" " + i for i in tokens]).strip()

#print(untokenize(["Hey",",","How","are","you","?"]))
#print(calc_bleu(["I","am","feeling","good"],["I","am","not","feeling","well"]))
#print(calc_bleu_many(["Hey",",","How","are","you","?"],[["You","are","well","?"],["How","you","doing","?"]]))
#print(tokenize("How are you?"))
