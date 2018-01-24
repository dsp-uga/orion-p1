import re
import json
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("P1PreProcess")
sc = SparkContext(conf = conf)

allStopWords = sc.textFile("file:///Users/.../data/stopwords/stopwords.txt")

# all stopwords to lowercase and stripping whitespaces
cleanedStopWords = list(map(lambda word: word.strip().lower(), allStopWords.collect()))

# broadcast stopwords
stopwords = sc.broadcast(cleanedStopWords)

# broadcast punctuations
punctuation = sc.broadcast(".,;:!?`'")

'''
def removePunctuation(wordsWithPunc):
    return re.compile(r'\W+', re.UNICODE).split(wordsWithPunc.lower().strip())
'''

combinedData = sc.textFile("file:///Users/.../data/X_train_*.txt")

'''
# get individual words
words = combinedData.flatMap(removePunctuation)
'''

# get individual words
words = combinedData.flatMap(lambda x: x.lower().split(" "))

words = words.map(lambda x: x.strip())

# filter out stopwords
words = words.filter(lambda x: x not in stopwords.value)

# counting the occurance of the filtered words
words = words.map(lambda x: (x, 1))

# add up the count of duplicate keys
words = words.reduceByKey(lambda x, y: x + y)

# sorting in reverse order based on counts
words = words.sortBy(lambda x: x[1] * -1).collect()

top40 = {}

for eachListPair in words:
    word = str(eachListPair[0])
    count = eachListPair[1]
    if(word):
        if('&quot;' in word):
            word = word.replace('&quot;','')
            if(word):
                top40[word] = count
        top40[word] = count

with open('p1_pre_process.json', 'w') as topWordsFile:
    json.dump(top40, topWordsFile, indent=2)

# Also couldn't figure out using saveAsHadoopFile()
