package com.guilherme.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

import java.io.*;
import java.util.List;


/**
 * Use this class to filter through GNIP tweets and extract only the tweet text
 * Use Spark for this for quicker processing. Best to do it by reading tweets from S3 bucket
 */
public class TweetFilter {

    private static String pathToFile = "data/tweets.json";
    private static String pathToSave = "data/reduced-tweets.txt";
    private boolean prettyPrint = false;
    // Whether or not to include time zone information
    private boolean includeTZ = false;


    public static void main(String[] args) throws IOException {
        TweetFilter tf = new TweetFilter();
        tf.createTweetText();
    }

    /**
     * Create the text file with one tweet per file. This way it's easy to train the word2vec model
     */
    private void createTweetText() {

        // create spark configuration and spark context
        SparkConf conf = new SparkConf()
                .setAppName("TweetFilter")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Initialize Spark Session
        SparkSession ssc = SparkSession
                            .builder().appName("TweetFilter")
                            .getOrCreate();

        // Using SQLContext read in the json for the tweets
        SQLContext sqlContext = ssc.sqlContext();
        Dataset<Row> tweets = sqlContext.read().json(pathToFile).cache();

        tweets.createOrReplaceTempView("tweetTable");

        if (prettyPrint) {
            System.out.println("------Tweet table Schema------");
            tweets.printSchema();
        }

        String query;
        // Get only the relevant features
        if (includeTZ) {
             query = "SELECT CONCAT(body, ' ', actor.location.displayName, ' ', actor.twitterTimeZone) " +
                    "AS text " +
                    "FROM tweetTable " +
                    "WHERE actor.location.displayName is not null AND actor.twitterTimeZone is not null";
        }
        else {
             query = "SELECT CONCAT(body, ' ', actor.location.displayName)" +
                    "AS text " +
                    "FROM tweetTable " +
                    "WHERE actor.location.displayName is not null";
        }

        Dataset<Row> results = sqlContext.sql(query);

        if (prettyPrint) {
            System.out.println("Some text examples");
            results.show();
        }

        List<Row> tweetList = results.collectAsList();
        // Write the each tweet, one per line, to a file that can then be used to train the word2vec model
        try {
            BufferedWriter outputWriter = new BufferedWriter(new FileWriter(pathToSave));
            for (Row row : tweetList) {
                String toWrite = row.toString();
                toWrite = toWrite.substring(1, toWrite.length() - 1);
                toWrite= toWrite.replace(";"," ").replace(":"," ").replace('"',' ')
                        .replace('-',' ').replace(',',' ').replace('.',' ')
                        .replace("!", " ").replace("\\n", " ").replace("\\r", " ");
                outputWriter.write(toWrite + "\n");
            }
            outputWriter.flush();
            outputWriter.close();
        } catch (IOException exception) {
            System.out.println("Bad path to save tweets to");
        }

        // This might be the best method of doing things, but unfortunately, it breaks files into multiple parts
        // System.out.println("Saving results to json file");
        // results.write().mode("append").json(pathToSave);
    }

}
