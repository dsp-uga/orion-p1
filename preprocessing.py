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

class preprocess:
    def __init__(self, x_train_path, x_test_path, y_train_path, spark_context, stopwords):
        self.x_train_path = x_train_path
        self.x_test_path = x_test_path
        self.y_train_path = y_train_path
        self.stopwords = stopwords
        self.sc = spark_context
        self.x_train = self.sc.textFile(self.x_train_path)
        self.x_test = self.sc.textFile(self.x_test_path)
        self.y_train = self.sc.textFile(self.y_train_path)
    def clean(self, row):
        pass
    def remove_stop_words(self, row):
        pass
    def stem(self, row):
        pass
    def get_vocab(self, row):
        pass
    def get_counts(self, row):
        pass
    def dupe(self, x_train, y_train):
        pass

if __name__ == '__main__':
    
