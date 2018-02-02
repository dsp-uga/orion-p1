# Project 1: Large Scale Document Classification

The current project aims to classify documents from Reuters New Corpus using a Naive Bayes Algorithm with Laplace smoothing (adding 1 to word count by default). Our goal will be to not only implement this algorithm but do so in a scalable manner while also trying to exceed the benchmark accuracy of 90%.

The flow of the program is as follows:
  1. Duplicate documents with multiple category classes and drop documents with no
  2. Pre-processing:
    - Remove numbers
    - Remove punctuation and special characters(not letters or numbers): and replace with
    - Remove stop words
    - Stem words
  3. Get list of every unique word across all documents to make a feature vector. (save)
  4. Get dataframe where each row is each document and each column is the words for train set. Amend with new words in test.
  5. +1 to all counts
  6. Calculate TF-IDF for all values in each document
  7. Calculate count table where rows are classes
  8. Calculate probabilities by class.

Predict:
  1. Use the generated probability table to calculate the document class for all the instances in the test set.
