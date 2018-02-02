# preprocessing class needs to take in input file and return the
# processed hadoop file that contains a count dataframe
# Author: Rajeswari Sivakumar
# The flow of the program is as follows:
#   1. Duplicate documents with multiple category classes
#   2. Pre-processing:
#     - Remove numbers
#     - Remove punctuation and special characters(not letters or numbers): and replace with
#     - Remove stop words
#     - Stem words
#   3. Get list of every unique word across all documents to make a feature vector. (save)
#   4. Get dataframe where each row is each document and each column is the words for train set. Amend with new words in test.
#   5. +1 to all counts
#   6. Calculate TF-IDF for all values in each document
#   7. Calculate count table where rows are classes
#   8. Calculate probabilities by class.
#
# Predict:
#   1. Use the generated probability table to calculate the document class for all the instances in the test set.

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, Row
# import pyspark.implicits._
import argparse
import string
import numpy as np
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
import nltk
from math import log


def rem_punct(row):
    """
    Arguments: Row that is a string.
    Do: Convert all to lowercase
        Remove leading and trailing punctuation including ",?/<>()-_+=!~*&$#@%^.
        Remove leading and trailing spaces
        Remove new line characters
    Return: List of processed words.
    """
    out = row.replace("\\t", " ")
    out = out.replace("quot;", "")
    out = out.replace("&amp;", " and ")
    out = out.replace("\\n", " ")
    out = out.replace("-", " ")
    out = out.replace("'", " ")
    out = out.replace("/", " ")
    out = out.replace(".", " ")
    out = out.lower().split(" ")
    out = [i.strip("\",?/<>()-_+=!~*&$#@%^.'") for i in out]
    out = [i for i in out if i is not '']
    return out

def rem_num(row):
    """
    Arguments: List of strings
    Do: Remove strings with numeric characters
    Return: List of words without numeric characters
    """
    out = [i for i in row if not any(j.isdigit() for j in i)]
    return out

def remove_stop_words(row, stopwords):
    """
    Take a list of words.
    Drop the words that are in the stopwords list.
    """

    out = [word for word in row if word not in stopwords]
    return out

def lemma_words(row):
    """
    Take a list of words.
    Change suffix of words from NLTK WordNetLemmatizer
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    out = [lemmatizer.lemmatize(word) for word in row]
    return out

def clean_row(row, stopwords):
    """
    Take a row and perform all cleaning functions on it
    """
    out = rem_punct(row)
    out = rem_num(out)
    out = lemma_words(out)
    out = remove_stop_words(out, stopwords)
    return out
def get_count(row, vocab):
    """
    Take a row and transform into a series of tuples: (word, count)
    """
    list_out = []
    for word in vocab:
        list_out.append(row.count(word)+1)
    return list_out
def get_test_count(row, vocab):
    """
    Take a row and transform into a series of tuples: (word, count)
    """
    list_out = []
    word_count =0
    for word in vocab:
        if row.count(word)>0:
            word_count = 1
        else:
            word_count = 0
        list_out.append(word_count)
    return list_out
def get_val(x):
    return x
def get_prob(row):
    out =[num/TOTAL_COUNTS.value[i] for i, num in enumerate(row)]
    return out
def predict(row):
    p_list = [CCAT_prob(row), ECAT_prob(row), GCAT_prob(row), MCAT_prob(row)]
    idx = np.argmax(p_list)
    if idx == 0:
        return 'CCAT'
    elif idx == 1:
        return 'ECAT'
    elif idx == 2:
        return 'GCAT'
    elif idx == 3:
        return 'MCAT'

def CCAT_prob(row):
    prob_vec = [log(p_q(val,p)) for val, p in zip(row, PW_CCAT.value)]
    return sum(prob_vec) + log(P_CCAT.value)
def ECAT_prob(row):
    prob_vec = [log(p_q(val,p)) for val, p in zip(row, PW_ECAT.value)]
    return sum(prob_vec) + log(P_ECAT.value)
def GCAT_prob(row):
    prob_vec = [log(p_q(val,p)) for val, p in zip(row, PW_GCAT.value)]
    return sum(prob_vec) + log(P_GCAT.value)
def MCAT_prob(row):
    prob_vec = [log(p_q(val,p)) for val, p in zip(row, PW_MCAT.value)]
    return sum(prob_vec) + log(P_MCAT.value)
def p_q(num, p):
    if num == 0:
        return 1-p
    else:
        return p
def get_paths():
    parser = argparse.ArgumentParser()
    # adapted from argparse code in p0_key.py (authored by Shannon Quinn)
    parser.add_argument("-x", "--train", required = True,
        help = "File containing training documents")
    parser.add_argument("-t", "--test", required = True,
        help = "File containing test documents")
    parser.add_argument("-y", "--labels", required = True,
        help = "Directory containing the books / text files.")

    args = vars(parser.parse_args())
    return args

def main():
    sc = SparkContext(master='local')

    # args = get_paths()

    x_train = sc.textFile("gs://uga-dsp/project1/train/X_train_large.txt")
    x_test= sc.textFile("gs://uga-dsp/project1/test/X_test_large.txt")
    y_train = sc.textFile("gs://uga-dsp/project1/train/y_train_large.txt")

    STOPWORDS = sc.broadcast(nltk.corpus.stopwords.words('english'))
    # Thanks @Chris Barrick for sharing the heads up about nltk stopwords

    clean_x_train = x_train.map(lambda row: clean_row(row, STOPWORDS.value))
    clean_x_test = x_test.map(lambda row: clean_row(row, STOPWORDS.value))
    y_cat = y_train.map(lambda row: row.split(","))

    train_vocab = clean_x_train.flatMap(lambda row: row).distinct()
    test_vocab = clean_x_test.flatMap(lambda row: row).distinct()
    VOCAB = sc.broadcast(train_vocab.union(test_vocab).distinct().collect())

    x_train_count = clean_x_train.map(lambda row: get_count(row, VOCAB.value))
    x_train_count = x_train_count.zip(y_cat).map(lambda row: ([row[0]], row[1])).flatMapValues(get_val)
    x_train_count = x_train_count.filter(lambda row: row[1]=='CCAT' or row[1]=='ECAT' or row[1]=='GCAT' or row[1]=='MCAT')
    x_train_count = x_train_count.map(lambda row: (row[1], row[0][0]))

    NUM_DOCS = x_train_count.count()
    NUM_CAT = x_train_count.countByKey()

    P_CCAT = sc.broadcast(NUM_CAT['CCAT']/NUM_DOCS)
    P_ECAT = sc.broadcast(NUM_CAT['ECAT']/NUM_DOCS)
    P_GCAT = sc.broadcast(NUM_CAT['GCAT']/NUM_DOCS)
    P_MCAT = sc.broadcast(NUM_CAT['MCAT']/NUM_DOCS)

    x_train_class = x_train_count.reduceByKey(lambda a,b: np.add(a,b).tolist())
    TOTAL_COUNTS = sc.broadcast(x_train_class.map(lambda row: row[1]).reduce(np.add).tolist())

    x_train_prob = x_train_class.map(lambda x: (x[0], get_prob(x[1])))
    PW_CCAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='CCAT').map(lambda row: row[1]).collect())
    PW_ECAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='ECAT').map(lambda row: row[1]).collect())
    PW_GCAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='GCAT').map(lambda row: row[1]).collect())
    PW_MCAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='MCAT').map(lambda row: row[1]).collect())

    x_test_count = clean_x_test.map(lambda row: get_test_count(row, VOCAB.value))

    predictions = x_test_count.map(lambda row: predict(row))

    predictions.foreach(print)

if __name__ == '__main__':
    main()
