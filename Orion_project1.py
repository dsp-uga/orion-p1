
# coding: utf-8

# # Orion Project 1

# ## Part 1 Preprocessing

# In[1]:

# Import packages
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import explode,col



# In[2]:

# Configure and create the Spark Context object 

conf = SparkConf().setAppName("Project1").setMaster("local[*]")
sc = SparkContext(conf=conf)


# In[3]:


# Read the documents locally, one record correspond to one doc
XTrainOri = sc.textFile("data_for_initial_local_training/X_train_vsmall.txt").cache()
YTrainOri = sc.textFile("data_for_initial_local_training/Y_train_vsmall.txt").cache()

# Load stopwords form the file provided in project0
stopWordsFile = sc.textFile("stopwords.txt")
stopWords = sc.broadcast(stopWordsFile.flatMap(lambda s: s.split()).collect())

# Broadcast punctuation variable
punctuation = sc.broadcast(".,:;'!?$&_-()%+1234567890*/=")


# To read the documents on GCP
#XTrainOri = sc.textFile("gs://uga-dsp/project1/train/X_train_vsmall.txt").cache()
#YTrainOri = sc.textFile("gs://uga-dsp/project1/train//Y_train_vsmall.txt").cache()


# In[4]:

def remove_puntuation_lowercase(s):
    out = (s[0].replace('quot','').replace('.','').strip(punctuation.value).lower(),s[1])
    return out
    
def remove_stopwords(s):
    stop_words = stopWords.value
    word,index = s
    if word not in stop_words:
        return (word,index) 

def preprocess_word_index(vector):
    """This function takes in a vector with index and words word,
        it generates (word,index) pair,
        where the index indicates which doc the word belongs to.
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


def combine_DF(DF1,DF2):
    """Define a function to combine two Dataframe by creating index"""
    index_DF1 = DF1.withColumn("columnindex", monotonically_increasing_id())
    index_DF2 =DF2.withColumn("columnindex", monotonically_increasing_id())
    out_RDD = index_DF1.join(index_DF2, index_DF1.columnindex == index_DF2.columnindex, 'inner').drop(index_DF2.columnindex).drop(index_DF1.columnindex)
    return out_RDD
    


# In[5]:

# split documents into words, remove punctuation, trainsform words to lower case

XTrain_preprocessed = XTrainOri.zipWithIndex().flatMap(preprocess_word_index)

#print(XTrain_preprocessed.take(10))


# ### Create dataframe for the features
# 
# <font color='red'>#NEED HELP HERE, it works but may slow the program down
# 
# 
# <br>Here I have not figured out a single function to count word per document
# <br> So I purely used map-reduce. First I created tuple ((word,index of document),1) and reduced by key in order to get the count for words belong to different document. Then I created Row object with dictionary in order to create dataframe </font>

# In[6]:

# Broadcast a vector that contains all word in the training set
# Will use this vector as column names for the final dataset
cols = sc.broadcast(XTrain_preprocessed.map(lambda s: s[0]).distinct().collect())


# In[7]:

# count the # of (word,index of document) pairs 

XTrainCPSC = XTrain_preprocessed.map(lambda s: (s,1)).reduceByKey( lambda a,b: a + b)
#print(XTrainCPSC.take(10))    


# In[8]:

# the structure of output: (index of document, dictionary of {word: count of word})

XTrainDict = XTrainCPSC.map(lambda x: (x[0][1],(x[0][0],x[1])))                            .groupByKey()                            .map(lambda x: (x[0],dict(x[1])))


#print(XTrainDict.take(1))


# In[9]:

# create row object for each document, put 0 if the word does not exist in the document

XTrainRow = XTrainDict.map(lambda x: Row(**{k:(x[1][k] if k in x[1].keys() else 0) for k in cols.value}))


# In[10]:

# Create the data frame for training features

sqlContext = SQLContext(sc)
XTrainDF = sqlContext.createDataFrame(XTrainRow)


# ### Create Dataframe for Y
# 
# Dataframe for the labels and combine it with the dataframe for the 
# features

# In[11]:

# Create the Ytrain dataframe

YTrainLabels = YTrainOri.map(lambda x: Row(filter_CAT(x)))
                    
YTrainDF = sqlContext.createDataFrame(YTrainLabels,['Label'])
#YTrainDF.head(5)


# In[12]:

# Combine the dataframe containing features and the dataframe containing label array
TrainDF_multi_label = combine_DF(XTrainDF,YTrainDF)


# In[13]:

# Explode by label column such that a row with multiple label is duplicated
TrainDF = TrainDF_multi_label.withColumn('Label',explode(TrainDF_multi_label.Label))


# In[14]:

#TrainDF.head(1)[:100]


# ## Part 2 Implement Naive Bayes Model

# ### Calculate P( Y = yk)
# 
# To calculate  P( Y = yk), we divide the frequency of each yk divived by the total count of Y

# In[15]:

YTrainStatRDD = YTrainDF.withColumn('Label',explode(YTrainDF.Label)).rdd


# In[16]:

# YStat is a frequency vector for yk
YStat = YTrainStatRDD.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y).collect()


# In[17]:

# YStatProb is P( Y = yk) 
YStatFreq = {x[0].asDict()['Label']:x[1] for x in YStat}
YStatProb = {x:YStatFreq[x]/sum(YStatFreq.values()) for x in YStatFreq}
print(YStatProb)


# ### Calculate P(xi|Y = yk)
# 
# <font color='red'># NEED HELP HERE, VERY SLOW PROGRAM
# <br>Here I tried to transform TrainDF into probability table that we discussed earlier. First I group the rows by same labels, and sum the word count for the same type of document. Then I tried to divide the sum by total word count for that perticular word, in order to get the probability</font>

# In[21]:

import time
start_time = time.time()

# group the rows by same labels, and sum the word count for the same type of document
exprs = {x: "sum" for x in cols.value}
TrainCount = TrainDF.groupBy('Label').agg(exprs)

print("It takes --- %s seconds --- to create the dataframe for features" % (time.time() - start_time))


# In[ ]:


"""Here the program runs forever"""
from pyspark.sql.functions import sum

for col_name in TrainCount.columns:
    if col_name != "Label":
        TrainCount = TrainCount.withColumn(col_name, col(col_name)/TrainCount.select(col(name)).rdd.map(lambda x: x[0]).reduce(lambda x,y: x+y))





