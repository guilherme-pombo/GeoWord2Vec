# GeoWord2Vec
Using twitter text and location data, build a word2vec model that can be queried for location specific words.

# Libraries used

The Twitter processing is done with Spark, the word2vec model build is done DeepLearning4j

# Running

First process GNIP twitter data using the

```
TwitterFilter.java
```

file. This will go through a .json file and output only the relevant data to training the word2vec model (tweet text, user location field and time zone (if you want)). Then train the word2vec model using:

```
GeoWord2Vec.java
```

This will save a .bin file in /output directory which contains the word2vec vectors. You can then query it for location indicative words. E.g. (for the tweets I trained it on):

```
Query: brasil
Results (top 4): [Rio, Janeiro, dilma, rousseff]
```

# Details

This word2vec is quite crude and it's just a potential idea to be used in indentifying the location of twitter users. By itself it's quite a weak indicator, but if used in combination with a Machine Learning model (e.g. CNNs) can lead to interesting results.
