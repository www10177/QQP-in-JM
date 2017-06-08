#!/usr/bin/env python3
from collections import Counter
import Parser
import pickle


def preprocessing(sentence):
    """
    Doing word preprocessing
    in : string
    out : list of words after preprocessing
    """
    parser = Parser.Parser()
    token = parser.tokenise(sentence)
    # return parser.removeStopWords(token)
    return token


def updateDict(dict1, dict2):
    """
    update dict2 to dict1
    same as dict1.update(dict2) but without overwriting
    """
    for k, v in dict2.items():
        if k in dict1:
            dict1[k] += v
        else:
            dict1[k] = v


class Corpus:
    """
    Members :
        data : list contains JMModel
        totalword : int which holds total words in this corpus
        worddict : dict that contains all {word : prob } in this corpus
    """
    # data = []

    def __init__(self, lamb):
        self.lamb = lamb
        self.count = 0
        self.totalword = 0
        self.worddict = {}

    def add(self, item, dataLocation):
        """
        add JMModel into corpus
        In : list of JMModels or JMModel
        """
        if hasattr(item, '__iter__'):
            # add list of JMModels
            for i in item:
                self.add(i)
        else:
            # add single JMModel
            # self.data.append(item)
            # with open('data/pkls/'+count+'.pkl') as data:
            item.save(dataLocation+str(self.count)+'.pkl')
            self.count += 1
            self.totalword += item.totalword
            updateDict(self.worddict, dict(item.wordCounter))

    def addWithExisted(self, item):
        self.count += 1
        self.totalword += item.totalword
        updateDict(self.worddict, dict(item.wordCounter))

    def prob(self, index, word, location, mode='prob'):
        """
        calculate prob of word or sentences
        In :
            Index : doing query on which JMModel
            word : sentence or single word which used to query
            mode : 'prob' will return only float
                   'list' will return list of each word probability
        Out : float or list controlled by mode para
        """

        if " " in word:
            if mode == 'list':
                result = []
                for w in preprocessing(word):
                    result.append(self.prob(index, w, location))
                    result.sort(key=lambda x: x[1], reverse=True)
                return result
            else:
                result = 1.0
                for w in preprocessing(word):
                    result *= self.prob(index, w, location)
                return result
        else:
            with open(location+str(index)+'.pkl', 'rb') as model:
                jmm = pickle.load(model)
                probdict = jmm.probdict
            # probdict = self.data[index].probdict
                if word in probdict:
                    # jm smoothing #
                    # lamb * P(w|D) + (1-lamb) * P(w|C) #
                    if mode == 'list':
                        return (word, self.lamb * probdict[word] + (
                            1-self.lamb)*self.worddict[word]/self.totalword)
                    else:
                        return self.lamb * probdict[word] + (
                            1-self.lamb)*self.worddict[word]/self.totalword

                elif word in self.worddict:
                    if mode == 'list':
                        return (
                            word, (1-self.lamb)*self.worddict[word]/self.totalword)
                    else:
                        return (1-self.lamb)*self.worddict[word]/self.totalword
                else:
                    print (word, ' is not in language model')
                    return 0
                del jmm, probdict


class JMModel:

    """
    class of language models
    members :
        token(list) : text tokenize into 1 word
        totalword(int) : total word count
        wordCounter(Counter) : Counter of every word count
        calProb(function) : the way to calculate(smoothing) probability
        probdict(dictionary) :  dict of word : prob
    """

    def __init__(self, text):
        """init and build dict"""
        token = preprocessing(text)
        self.totalword = len(token)
        self.wordCounter = Counter(token)
        calProb = lambda value: float(value) / self.totalword
        self.probdict = {
            name: calProb(value) for name,
            value in self.wordCounter.items()}

    def __del__(self):
        del self.wordCounter
        del self.totalword
        del self.probdict

    def save(self, location):
        with open(location, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)

    def wordProb(self, word):
        """
        calculate prob of word
        in : string
        """
        if word in self.probdict:
            return self.probdict[word]
        else:
            return 0


if __name__ == '__main__':
    with open('./test/0.pkl', 'rb')as f:
        p = pickle.load(f)
        print (p.probdict)
