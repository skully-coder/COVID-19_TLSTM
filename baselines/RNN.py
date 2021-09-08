import pickle
import sys
import numpy as np
import math
import tensorflow.compat.v1 as tf
print(tf.__version__)
# from tensorflow.models.RNNModel import RNNModel
# from tf.nn.rnn_cell import BasicLSTMCell, LSTMCell, GRUCell, RNNCell, BasicRNNCell, DropoutWrapper, MultiRNNCell
tf.disable_v2_behavior()

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj


def RNNModel(x, weights, biases):
    x = tf.unstack(x, 1)
    lstm_cell =tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.nn.rnn_cell.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out'])+biases['out']


if __name__ == "__main__":
    # train data
    path = 'BatchData'
    path_string = path + '/TrainData.seqs'
    data_train_batches = load_pkl(path_string)

    path_string = path + '/TrainLabel.seqs'
    labels_train_batches = load_pkl(path_string)
    number_train_batches = len(data_train_batches)
    input_dim = np.array(data_train_batches[0]).shape[2]
    output_dim = np.array(labels_train_batches[0]).shape[1]

    print("Train data is loaded!")

    path_string = path + '/TestData.seqs'
    data_test_batches = load_pkl(path_string)

    path_string = path + '/TestLabel.seqs'
    labels_test_batches = load_pkl(path_string)

    number_test_batches = len(data_test_batches)

    print("Test data is loaded!")

    training_rate = 0.001
    training_iters = 100000
    display_step = 10

    n_hidden = 64
    n_classes = 2
    tf.disable_eager_execution()
    x = tf.placeholder("float", [None, None, input_dim])
    y = tf.placeholder("float", [None, n_classes])

    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    pred = RNNModel(x, weights, biases)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=training_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuaracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step*number_train_batches < training_iters:
            for i in range(number_train_batches):
                batch_x, batch_y = data_train_batches[i], labels_train_batches[i]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if step % display_step == 0:
                    acc = sess.run(accuaracy, feed_dict={
                                   x: batch_x, y: batch_y})
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(step * number_train_batches) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))
                step += 1
