from textrank4zh import TextRank4Keyword
import jieba
print([i for i in jieba.cut('清朝灭亡了',HMM=True)])
tr4w = TextRank4Keyword()
text = '测试'
tr4w.analyze(text=text, lower=True, window=2)
for item in tr4w.get_keywords(20, word_min_len=1):
    print(item.word, item.weight)