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
  6. Calculate count table where rows are classes
  7. Calculate probabilities by class.

Predict:
  1. Use the generated probability table to calculate the document class for all the instances in the test set.


While we are able run our code successfully with the smaller `vsmall` data, we are unable to make a complete run on Google Cloud. We are unable to diagnose the errors as they seem to occur at various stage and are inconsistent despite not changing out code in significant way. We hope to examine this further soon.
