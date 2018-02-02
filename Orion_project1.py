# Import packages
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import Row
import time
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import explode,col
from pyspark.sql.functions import sum
from collections import Counter
#from nltk.stem import WordNetLemmatizer
import math

    
def remove_puntuation_lowercase(s):
    """This function removes puntuations"""
    out = s[0].replace("\\t", "")
    out = out.replace("quot;", "")
    out = out.replace("&amp;", " and ")
    out = out.replace("\\n", "")
    out = out.replace("-", "")
    out = out.replace("'", "")
    out = out.replace("/", "")
    out = out.replace(".", "")
    out = (out.replace(')','').strip(punctuation.value).lower(),s[1])
    return out
    
def remove_stopwords(s):
    """This function removes stopwords"""
    stop_words = stopWords.value
    word,index = s
    if word not in stop_words:
        word = lemma_words(word)
        return (word,index) 
    
def lemma_words(word):
    """
    Take a list of words.
    Change suffix of words from NLTK WordNetLemmatizer
    """
    #lemmatizer = WordNetLemmatizer()
    #out = lemmatizer.lemmatize(word)
    return word

def preprocess_word_index(vector):
    """This function takes in a vector with index and words,
        it generates (word,index) pair,
        where the index indicates which doc(or class) the word belongs to.
        The function also removes punctuation, stopwords and letter case"""
    splited_vector = list(map(lambda e: (e, vector[1]), vector[0].split()))
    new_vector = []
    for item in splited_vector:
        item = remove_puntuation_lowercase(item)
        item = remove_stopwords(item)
        if item and item[0]:
            new_vector.append(item)
    return new_vector


def filter_CAT(list):
    """Define a function such that the lable array only contains those end with 'CAT'"""
    list = list.split(',')
    outList = [x for x in list if x.endswith('CAT') ]
    return outList


def calculate_prob(list_word,vocab):
    """This function takes in a list of word
        It calculates the probability p(yi|x) 
        return a vector containing the conditional probability of the four class
        if_tfidf is a bolean parameter indicates if we use tfidf method to filter words"""
    prob = {'ECAT':0,'CCAT':0, 'GCAT':0, 'MCAT':0}
    for label in labels:
        log_class_priors = (math.log(YStatProb.value[label]))
        for word in list_word:
            if word in vocab.value and word in dict(Train_prob_dict.value)[label].keys():
                word_prob = dict(Train_prob_dict.value)[label][word]
                prob[label] = prob[label] + math.log(word_prob)   
        prob[label] = log_class_priors + prob[label]
    
    return prob
            

    
def classify(prob_dict):
    """This function takes in a vector of probability
        return a class name which has the highest probability"""
    sum = 0
    out_list = []
    out = max(prob_dict, key=(lambda key: prob_dict[key]))
    out_list.append(out)
    return out_list
        

def extract_vocab(Xtrain):
    """This function takes in an RDD of original training data for x
        return a vector containing all words of the training set"""
    XTrainOri_preprocessed = Xtrain.zipWithIndex().flatMap(preprocess_word_index)
    vocab_train = XTrainOri_preprocessed.map(lambda x: x[0]).distinct().collect()
    vocab = vocab_train
    return vocab


def extend_prob_dict(x,vocab,wordcount_bylabel):
    """This function calculates the prbability of xk given a class yi
        It takes in a dictionary of {word:word_count} and calculate the probability
        It adds one to both the denominator and numerator if the word does not exists in the class
        if_tfidf is a bolean parameter indicates if we use tfidf method to filter words"""
    
    for key in vocab.value:
        if key not in x[1]:
            x[1][key] = 1.0/(wordcount_bylabel.value[x[0]] + len(vocab.value))  
        else:
            x[1][key] = float(x[1][key] + 1)/(wordcount_bylabel.value[x[0]] + len(vocab.value))
    return (x[0],x[1])


def tf_idf(XTrainOri,num_word_document):
    """This function takes in the RDD for the original X_Train file
        It creates a vector of word that has the highest tf-idf for each document"""
    XTrain_index = XTrainOri.zipWithIndex().flatMap(preprocess_word_index)
    num_document = sc.broadcast(XTrainOri.count())
    XTrain_dict_index = XTrain_index.map(lambda s: (s,1))\
                            .reduceByKey( lambda a,b: a + b)\
                            .map(lambda x: (x[0][1],(x[0][0],x[1])))\
                            .groupByKey()\
                            .map(lambda x: (x[0],dict(x[1]))).cache()
    # Calculate IDF score for each word

    sub_IDF = XTrain_index.groupByKey()\
                                .map(lambda x: (x[0],math.log(float(num_document.value)/len(x[1]))))\
                                .collect()
    
            
    print(dict(sub_IDF[:10]))
    sub_IDF = sc.broadcast(dict(sub_IDF))
    
    sub_TFIDF = XTrain_dict_index.map(lambda x:  Counter({d:x[1][d]*sub_IDF.value[d] for d in x[1]}).most_common(num_word_document))
    col_TFIDF = sub_TFIDF.flatMap(lambda x: x).map(lambda x: x[0]).distinct().collect()
    return col_TFIDF
    
def calculate_proby(YTrainOri):
    """This function takes in the RDD for the original Y_Train file
        Output of the funtion is a dictionary of prior probabilitys for each label"""
    YTrainLabels = YTrainOri.map(lambda x: filter_CAT(x))
    YTrainRDD = YTrainLabels.flatMap(lambda x: x)
    # YStat is a frequency vector for yk
    YStat = YTrainRDD.map(lambda x: (x,1))\
            .reduceByKey(lambda x,y: x+y)\
            .collect()
    # YStatProb is P( Y = yk) 
    YStatFreq = dict(YStat)
    YStatProb = {x:YStatFreq[x]/math.fsum(YStatFreq.values()) for x in YStatFreq}
    return YStatProb

def combine_XY(XTrainOri,YTrainOri):
    """This function takes in the RDDs for the original X_Train and Y_Train file
        Out put an RDD that combine the two RDD by index, and preprocess the words/class
        Each row in the output RDD has the format (word,class_of_word)"""
    YTrainLabels = YTrainOri.map(lambda x: filter_CAT(x))
    YTrainLabels_index = YTrainLabels.zipWithIndex().map(lambda x: (x[1],x[0]))
    XTrainOri_index = XTrainOri.zipWithIndex().map(lambda x: (x[1],x[0]))

    Train_preprocessed = XTrainOri_index.join(YTrainLabels_index).map(lambda x: x[1]).flatMapValues(lambda x: x).flatMap(preprocess_word_index)
    return Train_preprocessed

def get_wordcount_by_label(Train_preprocessed):
    """This function takes in the RDD for preprocessed X_train data
        It calculates the word count for specific class
        The output is a dictionary of {class_name:total_word_count}"""
    XTrain_wordcountbylabel = Train_preprocessed.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).map(lambda x :(x[0][1],x[1]))
    wordcount_bylabel = dict(XTrain_wordcountbylabel.reduceByKey(lambda x,y: x+y).collect())
    return wordcount_bylabel

def fit(XTrainOri,Train_preprocessed,if_tfidf):
    if not if_tfidf:
        # Get the total word count for different classes                
        wordcount_bylabel = sc.broadcast(get_wordcount_by_label(Train_preprocessed))
    
        # Get all the words that appeared in training set
        vocab = sc.broadcast(extract_vocab(XTrainOri))
    else:
        # Get a vector of words with high tf_idf score
        vocab = sc.broadcast(tf_idf(XTrainOri,3))
        
        # Get the total word count for different classes                
        wordcount_bylabel = sc.broadcast(get_wordcount_by_label(Train_preprocessed.filter(lambda x: x[0] if x[0] in vocab.value  else None)))
    
    print("The current length of vocab vector: " ,len(vocab.value))
   
    # Get the probability of all p(xk|yi)
    Train_prob_dict_full = XTrainDict.map(lambda x: extend_prob_dict(x, vocab, wordcount_bylabel))
    return (Train_prob_dict_full.collect(),vocab)
    
if __name__ == '__main__':
    
    # Configure and create the Spark Context object 

    conf = SparkConf().setAppName("Project1").setMaster("yarn-client")
    sc = SparkContext(conf=conf)


    # Read the documents locally, one record correspond to one doc
    #XTrainOri = sc.textFile("data_for_initial_local_training/X_train_vsmall.txt")
    #YTrainOri = sc.textFile("data_for_initial_local_training/Y_train_vsmall.txt")
    #XTestOri = sc.textFile("data_for_initial_local_training/X_test_vsmall.txt")
    #YTestOri = sc.textFile("data_for_initial_local_training/Y_test_vsmall.txt")

    # Load stopwords form the file provided in project0
    stopWordsFile = sc.textFile("gs://irene024081/stopwords.txt")
    stopWords = sc.broadcast(stopWordsFile.flatMap(lambda s: s.split()).collect())

    # Broadcast punctuation variable
    punctuation = sc.broadcast(".,:;'!?$&_-()%+1234567890*/=\,?/<>()\"-_+=!~*&$#@%^.'")

    labels = ['ECAT','CCAT', 'GCAT', 'MCAT']


    # To read the documents on GCP
    XTrainOri = sc.textFile("gs://uga-dsp/project1/train/X_train_large.txt")
    YTrainOri = sc.textFile("gs://uga-dsp/project1/train/y_train_large.txt")
    XTestOri = sc.textFile("gs://uga-dsp/project1/test/X_test_large.txt")
    
    # Get the prior probability for Y
    YStatProb = sc.broadcast(calculate_proby(YTrainOri))
    print(YStatProb.value)

    Train_preprocessed = combine_XY(XTrainOri,YTrainOri)
    print(Train_preprocessed.take(10))

    # calculate the word count for a specific word per class.
    # The output of each row as (class,{word:wordcount})
    XTrainDict = Train_preprocessed.map(lambda s: (s,1))\
                                .reduceByKey( lambda a,b: a + b)\
                                .map(lambda x: (x[0][1],(x[0][0],x[1])))\
                                .groupByKey()\
                                .map(lambda x: (x[0],dict(x[1])))

    
    

    Train_prob_dict,vocab = sc.broadcast(fit(XTrainOri,Train_preprocessed,True)[0]),(fit(XTrainOri,Train_preprocessed,True)[1])
    
    # Check if sum(P(xi|Y = yk)) is 1

    for i in range(4):
        print(math.fsum(Train_prob_dict.value[i][1].values()))

    #print(Train_prob_dict.value)
        
    

    # Preprocess the test dataset
    XTestOri_preprocessed = XTestOri.zipWithIndex().flatMap(preprocess_word_index)

    #Fit the data using naive bayes model and make predictions
    start_time = time.time()
    predictedY = XTestOri_preprocessed.map(lambda x: (x[1],x[0])).groupByKey()\
                            .map(lambda x: calculate_prob(x[1],vocab))\
                            .map(lambda x: classify(x)).collect()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    print(predictedY)

    # write output
    with open('out.txt', 'a') as out_file:
        for s in predictedY:
            out_file.write(str(s) + '\n')
        
