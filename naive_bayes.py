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
from nltk.stem import WordNetLemmatizer
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
    lemmatizer = WordNetLemmatizer()
    out = [lemmatizer.lemmatize(word) for word in row]
    return out

def clean_row(row):
    """
    Take a row and perform all cleaning functions on it
    """
    out = rem_punct(row)
    out = rem_num(out)
    out = lemma_words(out)
    out = remove_stop_words(out, STOPWORDS.value)
    return out
def get_count(row):
    """
    Take a row and transform into a series of tuples: (word, count)
    """
    list_out = []
    for word in VOCAB.value:
        list_out.append(row.count(word)+1)
    return list_out
def get_test_count(row):
    """
    Take a row and transform into a series of tuples: (word, count)
    """
    list_out = []
    word_count =0
    for word in VOCAB.value:
        if row.count(word) >0:
            word_count = 1
        else
            word_count = 0
        list_out.append(word_count)
    return list_out
def get_val(x):
    return x
def get_prob(row):
    out =[num/TOTAL_COUNTS.value[i] for i, num in enumerate(row)]
    return out
def predict(row):
    pass
def CCAT_prob(row):
    prob_vec = [ for val, p in zip(row,PW_CCAT)]
    pass
def ECAT_prob(word):
    pass
def GCAT_prob(word):
    pass
def MCAT_prob(word):
    pass
def p_q(num, p):
    if num == 0:
        return 1-p
    else
        return p
def get_paths():
    pass

if __name__ == '__main__':
    sc = SparkContext(master = 'local')
    spark = SparkSession.builder \
          .appName("Learning Apach Spark") \
          .config("spark.some.config.option", "some-value") \
          .getOrCreate()

    x_train = sc.textFile("./data/X_train_vsmall.txt")
    x_test= sc.textFile("./data/X_test_vsmall.txt")
    y_train = sc.textFile("./data/y_train_vsmall.txt")
    stopwords_path = "stopwords.txt"

    with open(stopwords_path) as s:
        sw = s.readlines()
    STOPWORDS = sc.broadcast([i.strip() for i in sw])

    clean_x_train = x_train.map(lambda row: clean_row(row))
    clean_x_test = x_test.map(lambda row: clean_row(row))
    y_cat = y_train.map(lambda row: row.split(","))
    STOPWORDS.unpersist()

    train_vocab = clean_x_train.flatMap(lambda row: row).distinct()
    test_vocab = clean_x_test.flatMap(lambda row: row).distinct()
    VOCAB = sc.broadcast(train_vocab.union(test_vocab).distinct().collect())

    x_train_count = clean_x_train.map(lambda row: get_count(row))
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
    PW_CCAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='CCAT').map(lambda row: row[1]))
    PW_ECAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='ECAT').map(lambda row: row[1]))
    PW_GCAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='GCAT').map(lambda row: row[1]))
    PW_MCAT = sc.broadcast(x_train_prob.filter(lambda row: row[1]=='MCAT').map(lambda row: row[1]))

    x_test_count = clean_x_test.map(lambda row: get_test_count(row))

    predictions = x_test_count.map(lambda row: predict(row))
