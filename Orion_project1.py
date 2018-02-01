# Import packages
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import explode,col
from pyspark.sql.functions import sum
from nltk.stem import WordNetLemmatizer
import math

# Configure and create the Spark Context object 

conf = SparkConf().setAppName("Project1").setMaster("local[*]")
sc = SparkContext(conf=conf)


# Read the documents locally, one record correspond to one doc
XTrainOri = sc.textFile("data_for_initial_local_training/X_train_vsmall.txt")
YTrainOri = sc.textFile("data_for_initial_local_training/Y_train_vsmall.txt")
XTestOri = sc.textFile("data_for_initial_local_training/X_test_vsmall.txt")
YTestOri = sc.textFile("data_for_initial_local_training/Y_test_vsmall.txt")

# Load stopwords form the file provided in project0
stopWordsFile = sc.textFile("stopwords.txt")
stopWords = sc.broadcast(stopWordsFile.flatMap(lambda s: s.split()).collect())

# Broadcast punctuation variable
punctuation = sc.broadcast(".,:;'!?$&_-()%+1234567890*/=\,?/<>()\"-_+=!~*&$#@%^.'")

labels = ['ECAT','CCAT', 'GCAT', 'MCAT']


# To read the documents on GCP
#XTrainOri = sc.textFile("gs://uga-dsp/project1/train/X_train_large.txt")
#YTrainOri = sc.textFile("gs://uga-dsp/project1/train/y_train_large.txt")
#XTestOri = sc.textFile("gs://uga-dsp/project1/test/X_test_large.txt")

def remove_puntuation_lowercase(s):
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
    lemmatizer = WordNetLemmatizer()
    out = lemmatizer.lemmatize(word)
    return out

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


def calculate_prob(list_word):
    prob = {'ECAT':0,'CCAT':0, 'GCAT':0, 'MCAT':0}
    for label in labels:
        log_class_priors = (math.log(YStatProb.value[label]))
        for word in list_word:
            if word in vocab.value:
                word_prob = dict(Train_prob_dict.value)[label][word]
                prob[label] = prob[label] + math.log(word_prob)
        prob[label] = log_class_priors + prob[label]
    return prob
            

    
def classify(prob_dict):
    sum = 0
    out_list = []
    #for key in prob_dict:
    #    sum += prob_dict[key]
    #mean = sum/4
    #print(mean)
    #for key in prob_dict:
    #    if prob_dict[key] > mean*(1+threshold):
    #        out_list.append(key)
    out = max(prob_dict, key=(lambda key: prob_dict[key]))
    out_list.append(out)
    return out_list
        

def extract_vocab(Xtrain):
    XTrainOri_preprocessed = Xtrain.zipWithIndex().flatMap(preprocess_word_index)
    vocab_train = XTrainOri_preprocessed.map(lambda x: x[0]).distinct().collect()
    vocab = vocab_train
    return vocab




# split documents into words, remove punctuation, trainsform words to lower case
YTrainLabels = YTrainOri.map(lambda x: filter_CAT(x))

YTrainLabels_index = YTrainLabels.zipWithIndex().map(lambda x: (x[1],x[0]))
XTrainOri_index = XTrainOri.zipWithIndex().map(lambda x: (x[1],x[0]))

Train_preprocessed = XTrainOri_index.join(YTrainLabels_index).map(lambda x: x[1]).flatMapValues(lambda x: x).flatMap(preprocess_word_index)

print(Train_preprocessed.take(10))

# create rdd for features

XTrainDict = Train_preprocessed.map(lambda s: (s,1))\
                            .reduceByKey( lambda a,b: a + b)\
                            .map(lambda x: (x[0][1],(x[0][0],x[1])))\
                            .groupByKey()\
                            .map(lambda x: (x[0],dict(x[1]))).cache()
print(XTrainDict.take(1))  

XTrain_wordcountbylabel = Train_preprocessed.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).map(lambda x :(x[0][1],x[1]))

wordcount_bylabel = sc.broadcast(dict(XTrain_wordcountbylabel.reduceByKey(lambda x,y: x+y).collect()))

def extend_prob_dict(x):
    for key in vocab.value:
        if key not in x[1]:
            x[1][key] = 1/(wordcount_bylabel.value[x[0]] + len(vocab.value))  
        else:
            x[1][key] = (x[1][key] + 1)/(wordcount_bylabel.value[x[0]] + len(vocab.value))
    return (x[0],x[1])


YTrainRDD = YTrainLabels.flatMap(lambda x: x)
# YStat is a frequency vector for yk
YStat = YTrainRDD.map(lambda x: (x,1))\
            .reduceByKey(lambda x,y: x+y)\
            .collect()
from pyspark.sql.functions import sum

# YStatProb is P( Y = yk) 
YStatFreq = dict(YStat)
YStatProb = sc.broadcast({x:YStatFreq[x]/math.fsum(YStatFreq.values()) for x in YStatFreq})
print(YStatProb.value)

vocab = sc.broadcast(extract_vocab(XTrainOri))

Train_prob_dict = sc.broadcast(XTrainDict.map(extend_prob_dict).collect())

# Check if sum(P(xi|Y = yk)) is 1
for i in range(4):
    print(math.fsum(Train_prob_dict.value[i][1].values()))
print(wordcount_bylabel.value)

import time

XTestOri_preprocessed = XTestOri.zipWithIndex().flatMap(preprocess_word_index)

start_time = time.time()
predictedY = XTestOri_preprocessed.map(lambda x: (x[1],x[0])).groupByKey()\
                        .map(lambda x: calculate_prob(x[1]))\
                        .map(lambda x: classify(x)).collect()
print("--- %s seconds ---" % (time.time() - start_time))
        
print(predictedY)