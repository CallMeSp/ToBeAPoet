from textrank4zh import TextRank4Keyword
import synonyms
import re
import numpy as np
import gensim
from zhdic import zh2Hans


def loadCorpus(filename):
    title_list = []
    poem_list = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            title = line.strip().split(':')[0]
            poem = [
                term for term in re.split('[，。]',
                                          line.strip().split(':')[1])
                if len(term) > 1
            ]
            # poem = line.strip().split(':')[1]
            title_list.append(title)
            poem_list.append(poem)
    return title_list, poem_list


def extractKeywordFromUser(sentence, targetNum):
    textrank = TextRank4Keyword()
    s = sentence
    textrank.analyze(text=s, lower=True, window=2)
    keywords = [
        item.word for item in textrank.get_keywords(targetNum, word_min_len=1)
    ]
    candidatewords = []
    candidatescores = []
    if len(keywords) < targetNum:
        for keyword in keywords:
            wordlist, scorelist = synonyms.nearby(keyword)
            candidatewords.extend(wordlist)
            candidatescores.extend(scorelist)
        sortedIndex = np.argsort([-i for i in candidatescores])
        sortedIndex = [
            sortedIndex[i] for i in range(len(sortedIndex))
            if not candidatescores[sortedIndex[i]] == 1
        ]
        for i in range(targetNum - len(keywords)):
            keywords.append(candidatewords[sortedIndex[i]])
    return keywords


def getProcecessedPoems(rawTitles, rawPoems):
    embedding_path = './data/token_vec_300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(
        embedding_path, binary=False)
    count = {}
    for step, poem in enumerate(rawPoems):
        for line in poem:
            for token in line:
                try:
                    model[token]
                except:
                    if token not in zh2Hans:
                        count[step] = step
                    else:
                        try:
                            model[zh2Hans[token]]
                        except:
                            count[step] = step
    print(len(count), '/', len(rawPoems))
    processed_poems = []
    processed_titles = []
    for i in range(len(rawPoems)):
        if i not in count:
            processed_titles.append(rawTitles[i])
            newpoem = []
            for line in rawPoems[i]:
                newline = ''
                for token in line:
                    if token not in zh2Hans:
                        newline += token
                    else:
                        newline += zh2Hans[token]
                newpoem.append(newline)
            processed_poems.append(newpoem)
    return processed_titles, processed_poems


def genProcessedData(filename):
    filepath = './data/' + filename
    titles, poems = loadCorpus(filepath)
    tt, pp = getProcecessedPoems(titles, poems)
    with open('./data/pro_' + filename, 'w+') as fw:
        for i in range(len(tt)):
            fw.write(tt[i] + ':' + '，'.join(pp[i]) + '\n')


if __name__ == '__main__':
    pass
