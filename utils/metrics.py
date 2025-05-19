import nltk
import nltk.translate.bleu_score
import pandas as pd
import numpy as np


def bleu_score(reference,test):
    return nltk.translate.bleu_score.sentence_bleu([reference], test)