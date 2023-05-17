import pandas as pd
from util import build_bag_of_words_features_filtered
import pickle
import numpy as np
import nltk
import string
from nltk.classify import NaiveBayesClassifier
import os


def load_model():
    file_path = "./thinking_feeling_model.pkl"

    if not os.path.exists(file_path):
        print("Trying to load model")
        data_set = pd.read_csv("./mbti_1.csv")

        all_posts = pd.DataFrame()

        types = np.unique(np.array(data_set["type"]))
        total = data_set.groupby(["type"]).count() * 50

        # Organizing data
        for j in types:
            temp1 = data_set[data_set["type"] == j]["posts"]
            temp2 = []
            for i in temp1:
                temp2 += i.split("|||")
            temp3 = pd.Series(temp2)
            all_posts[j] = temp3

        useless_words = nltk.corpus.stopwords.words("english") + list(
            string.punctuation
        )

        ##### Introverted and Extroverted

        # Features for the bag of words model
        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = (
                temp1.dropna()
            )  # not all the personality types have same number of files
            if "I" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "introvert")
                        for i in temp1
                    ]
                ]
            if "E" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "extrovert")
                        for i in temp1
                    ]
                ]

        split = []
        for i in range(16):
            split += [len(features[i]) * 0.8]
        split = np.array(split, dtype=int)

        train = []
        for i in range(16):
            train += features[i][: split[i]]

        IntroExtro = NaiveBayesClassifier.train(train)
        with open("intro_extro_model.pkl", "wb") as file:
            pickle.dump(IntroExtro, file)

        #### Intution and Sensing

        # Features for the bag of words model
        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = (
                temp1.dropna()
            )  # not all the personality types have same number of files
            if "N" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "Intuition")
                        for i in temp1
                    ]
                ]
            if "E" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "Sensing")
                        for i in temp1
                    ]
                ]

        train = []
        for i in range(16):
            train += features[i][: split[i]]

        IntuitionSensing = NaiveBayesClassifier.train(train)
        with open("intuition_sensing_model.pkl", "wb") as file:
            pickle.dump(IntuitionSensing, file)

        #### Thinking Feeling
        # Features for the bag of words model
        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = (
                temp1.dropna()
            )  # not all the personality types have same number of files
            if "T" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "Thinking")
                        for i in temp1
                    ]
                ]
            if "F" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "Feeling")
                        for i in temp1
                    ]
                ]

        train = []
        for i in range(16):
            train += features[i][: split[i]]

        ThinkingFeeling = NaiveBayesClassifier.train(train)
        with open("thinking_feeling_model.pkl", "wb") as file:
            pickle.dump(ThinkingFeeling, file)

        #### Judging Perceiving
        # Features for the bag of words model
        features = []
        for j in types:
            temp1 = all_posts[j]
            temp1 = (
                temp1.dropna()
            )  # not all the personality types have same number of files
            if "J" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "Judging")
                        for i in temp1
                    ]
                ]
            if "P" in j:
                features += [
                    [
                        (build_bag_of_words_features_filtered(i), "Percieving")
                        for i in temp1
                    ]
                ]

        train = []
        for i in range(16):
            train += features[i][: split[i]]

        JudgingPercieiving = NaiveBayesClassifier.train(train)
        with open("judging_perceiving_model.pkl", "wb") as file:
            pickle.dump(JudgingPercieiving, file)
    else:
        print("using saved model")
