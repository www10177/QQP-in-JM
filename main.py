#!/usr/bin/env python3
# from pandas import read_csv
import pandas
import jmlm
import os
# import numpy as np
# import nltk


##############################
#  Configs : data locations  #
##############################
datalocation = './data/'
trainfile = 'test.csv'
lamb = 0.5
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
        print("Adding Questions...")
        print("Progress : {:.2%} Imported : ".format(
                  float(index) / totalcount) + str(index))
        corpus.add(jmlm.JMModel(i[0]))
        corpus.add(jmlm.JMModel(i[1]))
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
        l.append(
            (i, 1 if corpus.prob(
                2*i, questions[i][1]) >= threshold else 0))
        print("Querying...")
        print(
            "Progress : {:.2%} Querying : ".format(
                float(i) /
                totalcount) +
         str(i))
    return l


if __name__ == "__main__":
    source = pandas.read_csv(datalocation + trainfile)
    questions = getQuestions(source)
    corpus = addAllQuestionsToCorpus(questions, lamb)
    # printAllProb(questions, corpus)
    d = pandas.DataFrame(
        data=probList(
            questions,
            corpus),
        columns=[
            'test_id',
             'is_duplicate'])
    d.to_csv('test.csv', index=False)
