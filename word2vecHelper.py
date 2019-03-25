from myDataUtil import getTraindata
import numpy as np
import gensim


def extract_character_vocab(data):
    special_words = ['<PAD>', '<UNK>', '<START>', '<END>']
    set_words = list(set(
        [token for line in data for token in line[0] if len(token) > 0] +
        [token for line in data for token in line[1] if len(token) > 0] +
        [token for line in data for token in line[2] if len(token) > 0]))
    id2voc = {idx: word for idx, word in enumerate(special_words + set_words)}
    voc2id = {word: idx for idx, word in id2voc.items()}
    return id2voc, voc2id


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [list(sentence) + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


def getbatches_modified(targets, keywords, pretexts, batch_size, pad_int):
    '''
       定义生成器，用来获取batch
       '''
    for batch_i in range(0, len(targets) // batch_size):
        start_i = batch_i * batch_size
        keywords_batch = keywords[start_i:start_i + batch_size]
        pretexts_batch = pretexts[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_keywords_batch = np.array(pad_sentence_batch(keywords_batch, pad_int))
        pad_pretexts_batch = np.array(pad_sentence_batch(pretexts_batch, pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        pretexts_lengths = []
        for pretext in pretexts_batch:
            pretexts_lengths.append(len(pretext))

        keywords_lengths = []
        for keyword in keywords_batch:
            keywords_lengths.append(len(keyword))
            # keywords_lengths.append(1)

        yield pad_targets_batch, pad_keywords_batch, pad_pretexts_batch, targets_lengths, keywords_lengths, pretexts_lengths


def genwordEmbedding(id2word={}):
    modelpath = './data/token_vec_300.bin'
    embeddingModel = gensim.models.KeyedVectors.load_word2vec_format(modelpath, binary=False).wv
    embeddingMatrix = []
    for i in range(len(id2word)):
        try:
            embeddingMatrix.append(embeddingModel[id2word[i]])
        except:
            embeddingMatrix.append(list(np.random.rand(300)))
    return embeddingMatrix


if __name__ == '__main__':
    genwordEmbedding(None)
