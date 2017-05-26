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
trainfile = 'small.csv'
lamb = 0.5
threshold = 0.000000005


def getQuestions(data):
    """
    a easy function to convert dataFrame to nparray of questions
    :input: dataframe of raw csv data
    :output: an nparray of question1 and question2 text
    """
    return data[['question1', 'question2']].fillna(value='').values


def convertQuery(string):
    return jmlm.preprocssing(string)


def addAllQuestionsToCorpus(questions, lamb):
    corpus = jmlm.Corpus(lamb)
    totalcount = questions.shape[0]
    for index, i in enumerate(questions):
        print("Adding Questions...")
        print("Progress : {:.2%}".format(float(index) / totalcount))
        print("Imported : %d" % index)
        os.system('clear')
        corpus.add(jmlm.JMModel(i[0]))
        corpus.add(jmlm.JMModel(i[1]))
    return corpus


def printAllProb(questions, corpus):
    for i in range(0, len(questions)):
        print ("index : ", i)
        print ('%.15f' % (corpus.prob(2*i, questions[i][1])))


def probList(questions, corpus):
    l = []
    totalcount = questions.shape[0]
    for i in range(0, len(questions)):
        l.append(
            (i, 1 if corpus.prob(
                2*i, questions[i][1]) >= threshold else 0))
        print("Doing Query...")
        print("Progress : {:.2%}".format(float(i) / totalcount))
        print("Querying : %d" % i)
        os.system('clear')
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
