import tensorflow as tf
import numpy as np

X = np.random.randn(4, 10, 8)
# The second example is of length 6
X[1, 6:] = 0
X_lengths = [9, 8, 10, 7]

fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
outputs, states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=fw_cell, cell_bw=bw_cell, dtype=tf.float64, sequence_length=X_lengths, inputs=X
)

output_fw, output_bw = outputs
states_fw, states_bw = states

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    states_shape = tf.shape(states)
    print(states_shape.eval())
    c_f, h_f = states_fw
    o_f = output_fw
    c_b, h_b = states_bw
    o_b = output_bw
    # print('states_fw\n', sess.run(states_fw))
    # print('c_f\n', sess.run(c_f))
    # print('h_f\n', sess.run(h_f))
    print('o_f\n', tf.shape(o_f).eval(), '\n', sess.run(o_f))
    print('of_chip', sess.run(o_f[:, 0]))
    # print('c_b\n', sess.run(c_b))
    # print('h_b\n', sess.run(h_b))
    # print('o_b\n', sess.run(o_b))
