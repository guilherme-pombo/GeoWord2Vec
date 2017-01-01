package com.guilherme.word2vec;

import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;


public class GeoWord2Vec {

    private String inputFilePath = "data/reduced-tweets.txt";
    private String modelFilePath = "output/word2vec.bin";

    public static void main(String[] args) throws IOException {

        GeoWord2Vec geoWord2Vec = new GeoWord2Vec();
        geoWord2Vec.train();

        Word2Vec word2VecModel = WordVectorSerializer.readWord2VecModel(new File(geoWord2Vec.modelFilePath));

        // If using multiple words pass in a collection
        List<String> s = Arrays.asList("Rio", "de", "Janeiro");
        List<String> empty = Collections.emptyList();

        Collection<String> list = word2VecModel.wordsNearest("brasil", 10);
        System.out.println("brasil: " + list);

        list = word2VecModel.wordsNearest("philippines" , 10);
        System.out.println("philippines: " + list);

    }

    private void train() throws IOException {
        SentenceIterator sentenceIterator = new FileSentenceIterator(new File(inputFilePath));
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .layerSize(300)
                .windowSize(5)
                .seed(42)
                .epochs(3)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .build();

        System.out.println("Training word2vec model...");
        vec.fit();

        System.out.println("Saving vectors to disk at path: " + modelFilePath);
        WordVectorSerializer.writeWordVectors(vec, modelFilePath);
    }
}
