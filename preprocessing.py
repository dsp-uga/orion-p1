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
from pyspark.sql import SparkSession
import argparse
from nltk import stem

class preprocess:
    def __init__(self, x_train_path, x_test_path, y_train_path, spark_context, stopwords_path):
        self.x_train_path = x_train_path
        self.x_test_path = x_test_path
        self.y_train_path = y_train_path
        self.sc = spark_context
        self.stopwords = self.sc.broadcast(open(stopwords_path, "r").readlines())
        self.x_train = self.sc.textFile(self.x_train_path).cache()
        self.x_test = self.sc.textFile(self.x_test_path).cache()
        self.y_train = self.sc.textFile(self.y_train_path).cache)()

    def rem_punct(self, row):
        """
        Arguments: Row that is a string.
        Do: Convert all to lowercase
            Remove leading and trailing punctuation including ",?/<>()-_+=!~*&$#@%^.
            Remove leading and trailing spaces
            Remove new line characters
        Return: List of processed words.
        """
        out = row.lower().strip(string.punctuation).split(" ")
        return out

    def rem_num(self, row):
        """
        Arguments: List of strings
        Do: Remove strings with numeric characters
        Return: List of words without numeric characters
        """
        out = [i for i in row if any(j.isdigit() for j in i)]
        return out

    def remove_stop_words(self, row):
        """
        Take a list of words.
        Drop the words that are in the stopwords list.
        """
        out = [word for word in row if word not in self.stopwords.value]
        return out

    def stem_words(self, row):
        """
        Take a list of words.
        Change suffix of words from stem list based on Stanford NLTK.
        """
        out = [stem(word) for word in row]
        return out

    def clean_row(self, row):
        out = self.rem_punct(row)
        out = self.rem_num(out)
        out = self.remove_stop_words(out)
        out = self.stem_words(out)
        return out

    def clean_train(self):
        out = self.x_train.map(lambda row: self.clean_row(row))
        out1 = out.collect()
        return out1
    def clean_test(self):
        out = self.x_test.map(lambda row: self.clean_row(row))

        return out
    def get_vocab(self, rdd):
        pass
    def get_counts(self, row):
        pass
    def dupe(self, x_train, y_train):
        pass

x_train_path = "./data/X_train_vsmall.txt"
x_test_path = "./data/X_test_vsmall.txt"
y_train_path = "./data/y_train_vsmall.txt"
stopwords_path = "stopwords.txt"
conf = SparkConf().setMaster("local").setAppName("orion-p1")
sc = SparkContext( conf = conf)
pp = preprocess(x_train_path, x_test_path, y_train_path, sc, stopwords_path)
cleaned_train = pp.clean_train()
print(cleaned_train)

    # temp = cleaned_train.collect()
    # print(temp)
    # cleaned_test = pp.clean_test()
