import logging
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import myDataUtil
from word2vecHelper import extract_character_vocab, get_batches, getbatches_modified, genwordEmbedding


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


def get_encoder_layer_modified(batch_keyword_ids, batch_pretext_ids, rnn_size, num_layers, keywords_length,
                               pretexts_length, vocab_size, encoding_embedding_size):
    # encoder_embedding
    encoder_embeddings = tf.Variable(WORDEMBEDDING)
    print('encoder use preTrained wordembedding_char')
    encoder_embed_keywords = tf.nn.embedding_lookup(encoder_embeddings, batch_keyword_ids)
    encoder_embed_pretexts = tf.nn.embedding_lookup(encoder_embeddings, batch_pretext_ids)

    #  rnn cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    with tf.variable_scope('forward'):
        keywords_fw_cell = get_lstm_cell(rnn_size)
        pretexts_fw_cell = get_lstm_cell(rnn_size)

    with tf.variable_scope('backward'):
        keywords_bw_cell = get_lstm_cell(rnn_size)
        pretexts_bw_cell = get_lstm_cell(rnn_size)
    key_fw_cells = tf.contrib.rnn.MultiRNNCell([keywords_fw_cell for i in range(num_layers)])
    pre_fw_cells = tf.contrib.rnn.MultiRNNCell([pretexts_fw_cell for i in range(num_layers)])
    key_bw_cells = tf.contrib.rnn.MultiRNNCell([keywords_bw_cell for i in range(num_layers)])
    pre_bw_cells = tf.contrib.rnn.MultiRNNCell([pretexts_bw_cell for i in range(num_layers)])

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
    print('!!!!!!!!', encoder_outputs, encoder_final_state)
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
    decoder_embeddings = tf.Variable(WORDEMBEDDING)
    print('decoder use preTrained wordembedding_char')
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
    sequence_length = 10
    return [word2id.get(word, word2id['<UNK>']) for word in text] + [
        word2id['<PAD>']] * (sequence_length - len(text))


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


def predict(input_sentence):
    # 输入一个单词
    # user_keywords = myDataUtil.extractKeywordFromUser(input_sentence, 4)
    user_keywords = ['树', '土', '黄花', '白云']
    print('keywords:', user_keywords)
    text = [source_to_seq(word) for word in input_sentence]
    checkpoint = "./model/trained_model_attention.ckpt"

    input_keywords_ids = tf.placeholder(tf.int32, [None, None], name='inputs_keywords')
    input_pretexts_ids = tf.placeholder(tf.int32, [None, None], name='inputs_pretexts')
    input_keywords_length = tf.placeholder(tf.int32, [None], name='kerwords_sequence_length')
    input_pretexts_length = tf.placeholder(tf.int32, [None], name='pretexts_sequence_length')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    target_sequence_length = tf.placeholder(
        tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(
        target_sequence_length, name='max_target_len')

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # 加载模型
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_keyword = loaded_graph.get_tensor_by_name('inputs_keywords:0')
        input_pretext = loaded_graph.get_tensor_by_name('inputs_pretexts:0')
        print(input_keyword)
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        print(logits)
        keywords_sequence_length = loaded_graph.get_tensor_by_name('kerwords_sequence_length:0')
        pretexts_sequence_length = loaded_graph.get_tensor_by_name('pretexts_sequence_length:0')
        print(keywords_sequence_length)
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        print(target_sequence_length)
        testpretexts = [[]]
        testpretexts_length = [0, 5, 11, 17]
        for i in range(4):
            answer_logits = sess.run(logits, {input_keyword: [text[i]] * batch_size,
                                              input_pretext: [source_to_seq(testpretexts[i])] * batch_size,
                                              target_sequence_length: [5] * batch_size,
                                              keywords_sequence_length: [len(user_keywords[i])] * batch_size,
                                              pretexts_sequence_length: [testpretexts_length[i]] * batch_size})[0]
            pad = word2id["<PAD>"]
            responseIds = [i for i in answer_logits if i != pad]
            responseWords = " ".join([id2word[i] for i in answer_logits if i != pad])
            if not i == 0:
                testpretexts.append(testpretexts[i] + ',' + responseWords)
            else:
                testpretexts.append(responseWords)
            print('step', i, ':', testpretexts)

        print('原始输入:', 'keyword:', user_keywords)
        print('Response : {}'.format(testpretexts[-1]))


if __name__ == '__main__':
    # 构造映射表
    traindatas, keywords, pretexts, curlines = myDataUtil.getTraindata('train-wujue.txt')
    id2word_path = './data/id2word.pkl'
    word2id_path = './data/word2id.pkl'
    if os.path.exists(id2word_path) and os.path.exists(word2id_path):
        print('use exist word2id&id2word dict')
        with open(word2id_path, 'rb') as fr:
            word2id = pickle.load(fr)
        with open(id2word_path, 'rb') as fr:
            id2word = pickle.load(fr)
    else:
        print('generate new word2id&id2word dict')
        id2word, word2id = extract_character_vocab(traindatas)
        with open(word2id_path, 'wb') as fw:
            pickle.dump(word2id, fw)
        with open(id2word_path, 'wb') as fw:
            pickle.dump(id2word, fw)
    WORDEMBEDDING = genwordEmbedding(id2word)
    print('word2id lenths:', len(word2id), 'wordembedding_shape', len(WORDEMBEDDING), ',', len(WORDEMBEDDING[0]))
    # 对字母进行转换
    keywords_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] for line in keywords]
    pretexts_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] for line in pretexts]
    curlines_int = [[word2id.get(letter, word2id['<UNK>']) for letter in line] + [word2id['<END>']] for line in
                    curlines]
    # 超参数
    # Number of Epochs
    epochs = 30
    # Batch Size
    batch_size = 256
    # RNN Size
    rnn_size = 512
    # Number of Layers
    num_layers = 2
    # Embedding Size
    encoding_embedding_size = 300
    decoding_embedding_size = 300
    # Learning Rate
    learning_rate = 0.001
    print('modified attention start....')
    predict('小草偷偷地从土里钻出来，嫩嫩的，绿绿的。园子里，田野里，瞧去，一大片一大片满是的。坐着，躺着，打两个滚，踢几脚球，赛几趟跑，捉几回迷藏。风轻悄悄的，草软绵绵的。')
    # train_attention_modified()
