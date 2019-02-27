import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import myDataUtil
from word2vecHelper import extract_character_vocab

logging.basicConfig(
    level=logging.NOTSET,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# 构造映射表
traindatas, keywords, pretexts, curlines = myDataUtil.getTraindata('train-wujue.txt')

id2word, word2id = extract_character_vocab(traindatas)
# 对字母进行转换
keywords_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] for line in keywords]
pretexts_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] for line in pretexts]
curlines_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] + [word2id['<EOS>']] for line in curlines]

# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 150
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 150
decoding_embedding_size = 150
# Learning Rate
learning_rate = 0.001


def get_inputs():
    '''
    :return:模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    return inputs, targets, target_sequence_length, max_target_sequence_length, source_sequence_length


def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    '''

    :param input_data: 输入tensor
    :param rnn_size: rnn隐层结点数量
    :param num_layers: 堆叠的rnn cell数量
    :param source_sequence_length: 源数据的序列长度
    :param source_vocab_size: 源数据的词典大小
    :param encoding_embedding_size: embedding的大小
    :return: encoder_output, encoder_state
    '''
    # encoder_embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    #  rnn cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=source_sequence_length,
                                                      dtype=tf.float32)
    return encoder_output, encoder_state


def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    add '<start>' and delete the last token of the sequence because '<end>' will not be sent to rnn
    '''
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<START>']), ending], 1)
    return decoder_input


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length,
                   max_target_sequence_length, encoder_state, decoder_input):
    '''
        构造Decoder层

        参数：
        - target_letter_to_int: target数据的映射表
        - decoding_embedding_size: embed向量大小
        - num_layers: 堆叠的RNN单元数量
        - rnn_size: RNN单元的隐层结点数量
        - target_sequence_length: target数据序列长度
        - max_target_sequence_length: target数据序列最大长度
        - encoder_state: encoder端编码的状态向量
        - decoder_input: decoder端输入
    '''
    # embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # rnn cell in decoder
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])
    # full connection
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # training decoder
    with tf.variable_scope("decode"):
        # 得到help对象
        # 说明：Decoder端用来训练的函数。
        # 这个函数不会把t-1阶段的输出作为t阶段的输入，而是把target中的真实值直接输入给RNN。
        # 主要参数是inputs和sequence_length。返回helper对象，可以作为BasicDecoder函数的参数。
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length, time_major=False)
        # build decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)
    # predicting decoder
    # share parameters with 'training'
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<START>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        # send the output of 't-1' to 't' as input
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
                                                                     target_letter_to_int['<END>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predicting_helper, encoder_state, output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder, impute_finished=True,
                                                                            maximum_iterations=max_target_sequence_length)
        return training_decoder_output, predicting_decoder_output


def seq2seq_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size, encoder_embedding_size, decoder_embedding_size, rnn_size,
                  num_layers):
    # get state of encoder
    _, encoder_state = get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                                         encoder_embedding_size)
    # the input of decoder which processed
    decoder_input = process_decoder_input(targets)
    training_decoder_output, predicting_decoder_output = decoding_layer()
