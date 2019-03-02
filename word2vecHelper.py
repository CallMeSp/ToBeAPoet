from myDataUtil import getTraindata
import numpy as np

def extract_character_vocab(data):
    special_words = ['<PAD>', '<UNK>', '<START>', '<END>',',']
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
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

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
if __name__ == '__main__':
    id2word, word2id = extract_character_vocab(getTraindata('train-wujue.txt')[0])
    print(word2id['扬'])
