#!/usr/bin/env python3
# from pandas import read_csv
import pandas
import jmlm
import gc
import pickle
import sys
import os
# import numpy as np
# import nltk


##############################
#  Configs : data locations  #
##############################
datalocation = './data/'
pklLocation = '../../../../pkls/'
# pklLocation = './data/pkls/'
trainfile = 'stringPOS_train.csv'
lamb = 0.7
threshold = 0.000000005


def getQuestions(data):
    """
    a easy function to convert dataFrame to nparray of questions
    input: dataframe of raw csv data
    output: an nparray of question1 and question2 text
    """
    return data[['question1', 'question2']].fillna(value='').values


def convertQuery(string):
    """
    Preprocessing query(tokenise, remove stopwords)
    Input: string
    Output: list of words
    """
    return jmlm.preprocssing(string)


def addAllQuestionsToCorpus(questions, lamb):
    """
    Input : list of questions which made from getQuestions
          : Lamb for float as lambda of jm smoothing
    Ouput : a corpus instance which defined in jmlm.py
    """
    corpus = jmlm.Corpus(lamb)
    totalcount = questions.shape[0]
    for index, i in enumerate(questions):
        if index % 10000 == 0:
            print("Adding Questions...")
            print("Progress : {:.2%} Imported : ".format(
                    float(index) / totalcount) + str(index))
        a = jmlm.JMModel(i[0])
        b = jmlm.JMModel(i[1])
        corpus.add(a, pklLocation)
        corpus.add(b, pklLocation)
        del a, b
        if index % 10000 == 0:
            gc.collect()
    return corpus


def __printAllProb(questions, corpus):
    """
    Print all word probability in corpus dict
    Usually used for debug of check corpus
    Input : courpus defined in jmlm.py
          : list of questions came from getQuestions function
    """
    for i in range(0, len(questions)):
        print ("index : ", i)
        print ('%.15f' % (corpus.prob(2*i, questions[i][1])))


def probList(questions, corpus):
    """
    Do query on all two pair quesions
    Input : courpus defined in jmlm.py
          : list of questions came from getQuestions function
    return:  list
    return example : [(0, 1), (1,0), (2, 1)]
    """

    l = []
    totalcount = questions.shape[0]
    for i in range(0, len(questions)):
        # Threshold
        # l.append(
            # (i, 1 if corpus.prob(
                # 2*i, questions[i][1]) >= threshold else 0))

        # Probability
        prob = corpus.prob(2*i, questions[i][1], pklLocation)
        l.append((i, prob))
        del prob
        if i % 10000 == 0:
            print("Querying...")
            print(
                "Progress : {:.2%} Querying : ".format(
                    float(i) /
                    totalcount) +
            str(i))
            gc.collect()

    return l


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ('Usage : python3 main.py -p(-q)')
        print ('-p to preprocess')
        print ('-q to query')
        print ('ex: python3 main.py -q')
    # preprocessing
    elif sys.argv[1] == '-p':
        source = pandas.read_csv(datalocation + trainfile)
        questions = getQuestions(source)
        corpus = addAllQuestionsToCorpus(questions, lamb)
        with open(pklLocation + 'corpus.pkl', 'wb') as corpusSave:
            pickle.dump(corpus, corpusSave, protocol=pickle.HIGHEST_PROTOCOL)
    elif sys.argv[1] == '-q':
        source = pandas.read_csv(datalocation + trainfile)
        questions = getQuestions(source)
        with open(pklLocation+'corpus.pkl', 'rb') as pkl:
            corpus = pickle.load(pkl)
        # __printAllProb(questions, corpus)
            print (corpus.worddict)
            d = pandas.DataFrame(
                data=probList(
                    questions,
                    corpus),
                columns=[
                    'test_id',
                    'is_duplicate'])
            d.to_csv('test.csv', index=False, float_format = '%.13f')
    elif sys.argv[1] == '-r':
        # re-generate corpus.pkl
        corpus = jmlm.Corpus(lamb)
        for loc, d, files in os.walk(pklLocation):
            for f in files:
                with open(loc+f, 'rb') as modelf:
                    model = pickle.load(modelf)
                    corpus.addWithExisted(model)
        with open(pklLocation+'corpus.pkl', 'wb') as f:
            pickle.dump(corpus,f)
    else:
        print ('add -p to preprocess')
        print ('add -q to query')
        print ('ex: python3 main.py -q')
