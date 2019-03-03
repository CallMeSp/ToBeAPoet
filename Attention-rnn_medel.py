import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import myDataUtil
from word2vecHelper import extract_character_vocab, get_batches, getbatches_modified


def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(
        tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(
        target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(
        tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def get_inputs_modified():
    '''
    模型输入tensor
    '''
    input_keywords_ids = tf.placeholder(tf.int32, [None, None], name='inputs_keywords')
    input_pretexts_ids = tf.placeholder(tf.int32, [None, None], name='inputs_pretexts')
    input_keywords_length = tf.placeholder(tf.int32, [None], name='kerwords_sequence_length')
    input_pretexts_length = tf.placeholder(tf.int32, [None], name='pretexts_sequence_length')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(
        tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(
        target_sequence_length, name='max_target_len')

    return input_keywords_ids, input_pretexts_ids, targets, learning_rate, target_sequence_length, \
           max_target_sequence_length, input_keywords_length, input_pretexts_length


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
    encoder_embeddings = tf.Variable(tf.random_uniform([source_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(encoder_embeddings, input_data)

    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    #  rnn cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    encoder_fw_cell = get_lstm_cell(rnn_size)
    encoder_bw_cell = get_lstm_cell(rnn_size)
    (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_state, encoder_bw_state) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_fw_cell, cell_bw=encoder_bw_cell, inputs=encoder_embed_input,
        sequence_length=source_sequence_length, dtype=tf.float32, time_major=False)
    # shape:[batch_size,max_length,rnn_size]
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    encoder_final_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), 1)
    encoder_final_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), 1)
    encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
    return encoder_outputs, encoder_final_state


def get_encoder_layer_modified(batch_keyword_ids, batch_pretext_ids, rnn_size, num_layers, keywords_length,
                               pretexts_length, vocab_size, encoding_embedding_size):
    # encoder_embedding
    encoder_embeddings = tf.Variable(tf.random_uniform([vocab_size, decoding_embedding_size]))
    encoder_embed_keywords = tf.nn.embedding_lookup(encoder_embeddings, batch_keyword_ids)
    encoder_embed_pretexts = tf.nn.embedding_lookup(encoder_embeddings, batch_pretext_ids)

    #  rnn cell
    def get_lstm_cell(rnn_size, cellname):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    with tf.variable_scope('forward'):
        keywords_fw_cell = get_lstm_cell(rnn_size, 'test1')
        pretexts_fw_cell = get_lstm_cell(rnn_size, 'test3')

    with tf.variable_scope('backward'):
        keywords_bw_cell = get_lstm_cell(rnn_size, 'test2')
        pretexts_bw_cell = get_lstm_cell(rnn_size, 'test4')

    with tf.variable_scope('keywords'):
        (keywords_fw_outputs, keywords_bw_outputs), (
            keywords_fw_state, keywords_bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=keywords_fw_cell, cell_bw=keywords_bw_cell, inputs=encoder_embed_keywords,
            sequence_length=keywords_length, dtype=tf.float32, time_major=False)
    with tf.variable_scope('pretexts'):
        (pretexts_fw_outputs, pretexts_bw_outputs), (
            pretexts_fw_state, pretexts_bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=pretexts_fw_cell, cell_bw=pretexts_bw_cell, inputs=encoder_embed_pretexts,
            sequence_length=pretexts_length, dtype=tf.float32, time_major=False)
    # outputs shape:[batch_size,keywords_max_length,2*rnn_size]
    keywords_outputs = tf.concat((keywords_fw_outputs, keywords_bw_outputs), 2)
    keywords_final_state_c = tf.concat((keywords_fw_state.c, keywords_bw_state.c), 1)
    keywords_final_state_h = tf.concat((keywords_fw_state.h, keywords_bw_state.h), 1)
    keywords_final_state = tf.contrib.rnn.LSTMStateTuple(c=keywords_final_state_c, h=keywords_final_state_h)
    # outputs_shape:[batch_size,pretexts_max_length,2*rnn_size]
    pretexts_outputs = tf.concat((pretexts_fw_outputs, pretexts_bw_outputs), 2)
    pretexts_final_state_c = tf.concat((pretexts_fw_state.c, pretexts_bw_state.c), 1)
    pretexts_final_state_h = tf.concat((pretexts_fw_state.h, pretexts_bw_state.h), 1)
    pretexts_final_state = tf.contrib.rnn.LSTMStateTuple(c=pretexts_final_state_c, h=pretexts_final_state_h)
    # concat the keywords output and the pretexts outputs
    encoder_outputs = tf.concat((keywords_outputs, pretexts_outputs), 1)
    encoder_final_state = tf.concat((keywords_final_state, pretexts_final_state), 1)
    return encoder_outputs, encoder_final_state


def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    add '<start>' and delete the last token of the sequence because '<end>' will not be sent to rnn
    '''
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<START>']), ending], 1)
    return decoder_input


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, source_sequence_length,
                   max_target_sequence_length, encoder_output, decoder_input):
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
    # training decoder
    with tf.variable_scope("decode"):
        # 得到help对象
        # 说明：Decoder端用来训练的函数。
        # 这个函数不会把t-1阶段的输出作为t阶段的输入，而是把target中的真实值直接输入给RNN。
        # 主要参数是inputs和sequence_length。返回helper对象，可以作为BasicDecoder函数的参数。
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # attention cell
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_output,
                                                                   memory_sequence_length=source_sequence_length)
        attenion_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=rnn_size)
        de_state = attenion_cell.zero_state(batch_size, dtype=tf.float32)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(attenion_cell, target_vocab_size)
        decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, training_helper, de_state,
                                                  tf.layers.Dense(target_vocab_size))
        # build decoder
        training_decoder_output, training_decoder_state, training_decoder_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder, swap_memory=True)
    # predicting decoder
    # share parameters with 'training'
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(
            tf.constant([target_letter_to_int['<START>']], dtype=tf.int32),
            [batch_size],
            name='start_tokens')
        # 它和TrainingHelper的区别在于它会把t-1下的输出进行embedding后再输入给RNN。
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_embeddings, start_tokens, target_letter_to_int['<END>'])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_output,
                                                                   memory_sequence_length=source_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=rnn_size)
        de_state = decoder_cell.zero_state(batch_size, dtype=tf.float32)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, target_vocab_size)
        decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, predicting_helper, de_state,
                                                  tf.layers.Dense(target_vocab_size))
        # build decoder
        predict_decoder_output, predict_decoder_state, predict_decoder_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder, swap_memory=True, impute_finished=True,
            maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predict_decoder_output


def seq2seq_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size, encoder_embedding_size, decoder_embedding_size, rnn_size,
                  num_layers):
    # get state of encoder
    encoder_output, encoder_state = get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length,
                                                      source_vocab_size,
                                                      encoder_embedding_size)
    # the input of decoder which processed
    decoder_input = process_decoder_input(targets, word2id, batch_size)
    training_decoder_output, predicting_decoder_output = decoding_layer(word2id, decoding_embedding_size,
                                                                        num_layers, rnn_size, target_sequence_length,
                                                                        source_sequence_length,
                                                                        max_target_sequence_length, encoder_output,
                                                                        decoder_input)
    return training_decoder_output, predicting_decoder_output


def seq2seq_model_modified(keywords_ids, pretexts_ids, targets, lr, target_sequence_length, max_target_sequence_length,
                           keywords_sequence_length, pretexts_sequence_length, source_vocab_size, target_vocab_size,
                           encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers):
    # get state of encoder
    encoder_output, encoder_state = get_encoder_layer_modified(keywords_ids, pretexts_ids, rnn_size, num_layers,
                                                               keywords_sequence_length, pretexts_sequence_length,
                                                               source_vocab_size,
                                                               encoder_embedding_size)
    # the input of decoder which processed
    decoder_input = process_decoder_input(targets, word2id, batch_size)
    source_sequence_length = keywords_sequence_length + pretexts_sequence_length
    # print(keywords_sequence_length)
    # print(pretexts_sequence_length)
    # print(source_sequence_length)
    training_decoder_output, predicting_decoder_output = decoding_layer(word2id, decoding_embedding_size,
                                                                        num_layers, rnn_size, target_sequence_length,
                                                                        source_sequence_length,
                                                                        max_target_sequence_length, encoder_output,
                                                                        decoder_input)
    return training_decoder_output, predicting_decoder_output


def source_to_seq(text):
    # 对源数据进行转换
    sequence_length = 15
    return [word2id.get(word, word2id['<UNK>']) for word in text] + [
        word2id['<PAD>']] * (sequence_length - len(text))


# Attention 版本的train
def train_attention():
    # 构造graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        # 获得模型输入

        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs(
        )
        training_decoder_output, predict_output = seq2seq_model(
            input_data, targets, lr, target_sequence_length,
            max_target_sequence_length, source_sequence_length,
            len(word2id), len(word2id),
            encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers)
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(
            predict_output.sample_id, name='predictions')
        masks = tf.sequence_mask(
            target_sequence_length,
            max_target_sequence_length,
            dtype=tf.float32,
            name='masks')
        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets,
                                                    masks)
            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)
            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                                for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    # 将数据集分割为train和validation
    train_source = keywords_int[50 * batch_size:]
    train_target = curlines_int[50 * batch_size:]
    # 留出一个batch进行验证
    valid_source = keywords_int[:50 * batch_size]
    valid_target = curlines_int[:50 * batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
        get_batches(valid_target, valid_source, batch_size,
                    word2id['<PAD>'],
                    word2id['<PAD>']))
    display_step = 50  # 每隔50轮输出loss
    checkpoint = "./model/trained_model_attention.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                                word2id['<PAD>'],
                                word2id['<PAD>'])):
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: sources_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths})

                if batch_i % display_step == 0:
                    # 计算validation loss
                    validation_loss = sess.run(
                        [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         source_sequence_length: valid_sources_lengths})

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(train_source) // batch_size,
                                  loss,
                                  validation_loss[0]))

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')  # 构造graph


# modified Attention 版本的train
def train_attention_modified():
    # 构造graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        # 获得模型输入
        input_keywords_ids, input_pretexts_ids, targets, lr, target_sequence_length, max_target_sequence_length, \
        input_keywords_length, input_pretexts_length = get_inputs_modified()
        training_decoder_output, predict_output = seq2seq_model_modified(
            input_keywords_ids, input_pretexts_ids, targets, lr, target_sequence_length,
            max_target_sequence_length, input_keywords_length, input_pretexts_length, len(word2id), len(word2id),
            encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers)
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(
            predict_output.sample_id, name='predictions')
        masks = tf.sequence_mask(
            target_sequence_length,
            max_target_sequence_length,
            dtype=tf.float32,
            name='masks')
        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets,
                                                    masks)
            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)
            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                                for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    # 将数据集分割为train和validation
    train_keywords = keywords_int[50 * batch_size:]
    train_pretexts = pretexts_int[50 * batch_size:]
    train_target = curlines_int[50 * batch_size:]
    # 留出一个batch进行验证
    valid_keywords = keywords_int[:50 * batch_size]
    valid_pretexts = pretexts_int[:50 * batch_size]
    valid_target = curlines_int[:50 * batch_size]
    (valid_targets_batch, valid_keywords_batch, valid_pretexts_batch, valid_targets_lengths, valid_keywords_lengths,
     valid_pretexts_length) = next(
        getbatches_modified(valid_target, valid_keywords, valid_pretexts, batch_size, word2id['<PAD>']))
    display_step = 50  # 每隔50轮输出loss
    checkpoint = "./model/trained_model_attention.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, epochs + 1):
            for batch_i, (targets_batch, keywords_batch, pretexts_batch, targets_lengths, batch_keywords_lengths,
                          batch_pretexts_lengths) in enumerate(
                getbatches_modified(train_target, train_keywords, train_pretexts, batch_size, word2id['<PAD>'])):
                _, loss = sess.run(
                    [train_op, cost],
                    {input_keywords_ids: keywords_batch,
                     input_pretexts_ids: pretexts_batch,
                     targets: targets_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     input_pretexts_length: batch_pretexts_lengths,
                     input_keywords_length: batch_keywords_lengths
                     })

                if batch_i % display_step == 0:
                    # 计算validation loss
                    validation_loss = sess.run(
                        [cost],
                        {input_keywords_ids: valid_keywords_batch,
                         input_pretexts_ids: valid_pretexts_batch,
                         targets: valid_targets_batch,
                         lr: learning_rate,
                         target_sequence_length: valid_targets_lengths,
                         input_keywords_length: valid_keywords_lengths,
                         input_pretexts_length: valid_pretexts_length})

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(train_target) // batch_size,
                                  loss,
                                  validation_loss[0]))

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')  # 构造graph


if __name__ == '__main__':
    # 构造映射表
    traindatas, keywords, pretexts, curlines = myDataUtil.getTraindata('train-wujue.txt')
    id2word, word2id = extract_character_vocab(traindatas)
    print('word2id lenths:', len(word2id))
    # 对字母进行转换
    keywords_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] for line in keywords]
    pretexts_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] for line in pretexts]
    curlines_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] + [word2id['<END>']] for line in
                    curlines]
    # 超参数
    # Number of Epochs
    epochs = 60
    # Batch Size
    batch_size = 128
    # RNN Size
    rnn_size = 196
    # Number of Layers
    num_layers = 1
    # Embedding Size
    encoding_embedding_size = 300
    decoding_embedding_size = 300
    # Learning Rate
    learning_rate = 0.001
    print('modified attention start....')
    train_attention_modified()
