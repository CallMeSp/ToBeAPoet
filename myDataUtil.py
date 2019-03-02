from textrank4zh import TextRank4Keyword
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import synonyms
import re
import numpy as np
import gensim
from zhdic import zh2Hans


def loadCorpus(filename):
    title_list = []
    poem_list = []
    with open(filename, 'r', encoding='utf8') as fr:
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
    with open('./data/pro_' + filename, 'w+', encoding='utf8') as fw:
        for i in range(len(tt)):
            fw.write(tt[i] + ':' + '，'.join(pp[i]) + '\n')


def genTrainData(filename):
    filepath = './data/' + filename
    titles, poems = loadCorpus(filepath)
    trainDatas = []
    count = {}
    for step, poem in enumerate(poems):
        try:
            for i in range(len(poem)):
                keyword = extractKeywordFromUser(poem[i], targetNum=1)[0]
                preText = ','.join(poem[:i])
                curLine = poem[i]
                trainDatas.append([keyword, preText, curLine])
                # print(keyword, '||', preText, '||', curLine)
        except:
            count[step] = 1
            print(step, '/', '16786')
    with open('./data/train-wujue.txt', 'w+', encoding='utf8') as fw:
        for line in trainDatas:
            fw.write('|'.join(line) + '\n')
    print(len(count), '/', len(poems))


def getTraindata(filename):
    filepath = './data/' + filename
    traindatas = []
    keywords = []
    pretexts = []
    curlines = []
    with open(filepath, 'r', encoding='utf8') as fr:
        for line in fr.readlines():
            line = line.strip().split('|')
            keywords.append(line[0])
            pretexts.append(line[1])
            curlines.append(line[2])
            traindatas.append([line[0], line[1], line[2]])
    return traindatas, keywords, pretexts, curlines


if __name__ == '__main__':
    genTrainData('pro_wujue-all.txt')
    pass
