#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/18/17 9:41 AM
# @Author  : Jackokie Zhao , jianqin
# @File    : modulation.py
# @Software: PyCharm

# This coding is the modulation of a paper(https://arxiv.org/pdf/1602.04105.pdf).
import os
import pickle
import  scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from scipy import  fftpack
from functools import partial


num_epoch = 10
batch_size = 256
learning_rate = 5e-6
train_ratio = 0.9
log_dir = './log/'
[height, width] = [2, 512]
num_channels = 1
num_kernel_1 = 64
num_kernel_2 = 16
hidden_units = 128
dropout = 0.5
num_classes = 5
train_show_step = 10
test_show_step = 100
seed = 'jackokie'
reg_val_l1 = 0.001
reg_val_l2 = 0.001
import math
is_adam=True



def lrelu(x, leak=0.05, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def rrelu(x, leak=-0.6, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def signal_IQ(signal):
    '''transfrom signal to IQ'''
    Q=np.zeros(signal.shape)
    for i in range(signal.shape[0]):
        Q[i,:]+=fftpack.hilbert(signal[i,:])
    IQ=np.hstack((signal,Q))
    return IQ

def load_data(data_path, input_shape):
    """ Load the original data for training...
    Parameters:
        data_path: The original data path.
        input_shape:
    Returns:
        train_data: Training data structured.
    """
    # load the original data.
    orig_data = pickle.load(open(data_path, 'rb'), encoding='iso-8859-1')

    # Get the set of snr & modulations
    mode_snr = list(orig_data.keys())
    mods, snrs = [sorted(list(set(x[i] for x in mode_snr))) for i in [0, 1]]
    mods.remove('AM-DSB')
    mods.remove('QAM16')
    mods.remove('AM-SSB')

    # Build the train set.
    samples = []
    labels = []
    samples_snr = []
    mod2cate = dict()
    cate2mod = dict()
    for cate in range(len(mods)):
        cate2mod[cate] = mods[cate]
        mod2cate[mods[cate]] = cate

    for snr in snrs:
        for mod in mods:
            samples.extend(orig_data[(mod, snr)])
            labels.extend(1000 * [mod2cate[mod]])
            samples_snr.extend(1000 * [snr])

    shape = [len(labels), height, width, 1]
    samples = np.array(samples).reshape(shape)
    samples_snr = np.array(samples_snr)
    labels = np.array(labels)
    return samples, labels, mod2cate, cate2mod, snrs, mods, samples_snr



def load_one_type(type):
    path=os.path.join('..','signal_data',type)
    print("read data from ",path)
    data=[]
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.mat':
            if type=='8PSKGray':
                signal=sio.loadmat(os.path.join(path,file))['x']
            else:
                signal = sio.loadmat(os.path.join(path, file))['s']

            signal=signal[:,0:-(signal.shape[1]%width)]
            signal=np.reshape(signal,[-1,width])
            if type in set(['8PSKGray','CPFSK']):
                IQ=np.hstack((signal.real,signal.imag))
            else :
                IQ=np.reshape(signal_IQ(signal),[-1,height,width,1])
            data.extend(IQ)
    return data

def load_types(types):
    mod2cate={}
    cate2mod={}
    samples=[]
    labels=[]

    for i in range(len(types)):
        cate2mod[i]=types[i]
        mod2cate[types[i]]=i
        temp_signal=load_one_type(types[i])
        samples.extend(temp_signal)
        labels.extend(len(temp_signal)*[i])

    shape = [len(labels), height, width, 1]
    samples = np.array(samples).reshape(shape)
    labels = np.array(labels)
    return samples, labels, mod2cate, cate2mod





def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Parameters:
        var: The parameter which should be summarize.
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def accuracy_compute(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels.
    Parameters:
        predictions: The prediction logits matrix.
        labels: The real labels of prediction data.
    Returns:
        accuracy: The predictions' accuracy.
    """
    with tf.name_scope('test_accuracy'):
        accu = 100 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]
        tf.summary.scalar('test_accuracy', accu)
    return accu


def conv(data, kernel_shape, activation, name, dropout=None, regularizer=None, reg_val=0):
    """ Convolution layer.
    Parameters:
        data: The input data.
        kernel_shape: The kernel_shape of current convolutional layer.
        activation: The activation function.
        name: The name of current layer.
        dropout: Whether do the dropout work.
        regularizer: Whether use the L2 or L1 regularizer.
        reg_val: regularizer value.
    Return:
        conv_out: The output of current layer.
    """
    if regularizer == 'L1':
        regularizer = layers.l1_regularizer(reg_val)
    elif regularizer == 'L2':
        regularizer = layers.l2_regularizer(reg_val)

    with tf.name_scope(name):
        # Convolution layer 1.
        with tf.variable_scope('conv_weights', regularizer=regularizer):
            conv_weights = tf.Variable(
                tf.truncated_normal(kernel_shape, stddev=0.1, dtype=tf.float32))
            variable_summaries(conv_weights)
        with tf.variable_scope('conv_bias'):
            conv_biases = tf.Variable(
                tf.constant(0.0, dtype=tf.float32, shape=[kernel_shape[3]]))
        with tf.name_scope('conv'):
            conv = tf.nn.conv2d(data, conv_weights, strides=[1, 1, 1, 1], padding='SAME')
        with tf.name_scope('activation'):
            conv_out = activation(tf.nn.bias_add(conv, conv_biases))
            variable_summaries(conv_out)
        if dropout is not None:
            with tf.name_scope('dropout'):
                conv_out = tf.nn.dropout(conv_out, dropout)

        return conv_out


def hidden(data, activation, name, hidden_units, dropout=None, regularizer=None, reg_val=None):
    """ Hidden layer.
    Parameters:
        data: The input data.
        activation: The activation function.
        name: The layer's name.
        hidden_units: Number of hidden_out units.
        dropout: Whether do the dropout job.
        regularizer: Whether use the L2 or L1 regularizer.
        reg_val: regularizer value.
    Return:
        hidden_out: Output of current layer.
    """
    if regularizer == 'L1':
        regularizer = layers.l1_regularizer(reg_val)
    elif regularizer == 'L2':
        regularizer = layers.l2_regularizer(reg_val)

    with tf.name_scope(name):
        # Fully connected layer 1. Note that the '+' operation automatically.
        with tf.variable_scope('fc_weights', regularizer=regularizer):
            input_units = int(data.shape[1])
            fc_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([input_units, hidden_units],
                                    stddev=0.1, dtype=tf.float32))
            variable_summaries(fc_weights)
        with tf.name_scope('fc_bias'):
            fc_biases = tf.Variable(
                tf.constant(0.0, dtype=tf.float32, shape=[hidden_units]))
            variable_summaries(fc_biases)
        with tf.name_scope('activation'):
            hidden_out = activation(tf.nn.xw_plus_b(data, fc_weights, fc_biases))
            variable_summaries(hidden_out)
        if dropout is not None:
            hidden_out = tf.nn.dropout(hidden_out, dropout)
        return hidden_out


def classier_mlp(signals,base,batch_Norm,reuse=False):
    len=int(math.log(int(width/base),2))
    size=[int(width/pow(2,i+1)) for i in range(len)]
    # size = [base * pow(2, i) for i in range(len)]
    # signals=tf.reshape(signals[:,:-1], [-1, number])

    signals\
        =tf.reshape(signals, [batch_size, width*height])
    print(signals.shape)
    with tf.variable_scope('classier') as scope:
        if reuse:
            scope.reuse_variables()

        for i in range(len):
            signals = tf.contrib.layers.fully_connected(signals, size[i], activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm if batch_Norm else None,scope="layer_{}".format(i))

        logit = tf.contrib.layers.fully_connected(signals, num_classes, activation_fn=lrelu,normalizer_fn=tf.contrib.layers.batch_norm if batch_Norm else None,scope="layer_last")
    return logit
def cnn_1_model(input_pl, activation=tf.nn.relu, dropout=None):
    """ CNN 2 Model in the paper.
    Parameters:
        input_pl: The input data placeholder.
        activation: The activation function.
        dropout: Whether use the dropout.
    Returns:
        logits: The model output value for each category.
     """
    kernel1 = [1, 5, 1, 64]
    kernel2 = [2, 3, 64, 32]
    conv1 = conv(input_pl, kernel1, activation, 'conv_1', dropout, 'L2', reg_val_l2)
    conv2 = conv(conv1, kernel2, activation, 'conv_2', dropout, 'L2', reg_val_l2)

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    flatten = tf.reshape(conv2, [batch_size, width * height * 16])

    hidden_1 = hidden(flatten, activation, 'hidden_1', 128, dropout, 'L1', reg_val_l1)
    logits = hidden(hidden_1, activation, 'hidden_2', num_classes)

    return logits


def cnn_2_model(input_pl, activation=tf.nn.relu, dropout=None):
    """ CNN 2 Model in the paper.
    Parameters:
        input_pl: The input data placeholder.
        activation: The activation function.
        dropout: Whether use the dropout.
    Returns:
        logits: The model output value for each category.
     """
    with tf.variable_scope('classier') as scope:
        kernel1 = [1, 6, num_channels, num_kernel_1]
        kernel2 = [2, 5, num_kernel_1, num_kernel_2]
        conv1 = conv(input_pl, kernel1, activation, 'conv_1', dropout)
        conv2 = conv(conv1, kernel2, activation, 'conv_2', dropout)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        flatten = tf.reshape(conv2, [batch_size, width * height * num_kernel_2])

        hidden_1 = hidden(flatten, activation, 'hidden_1', 256, dropout)
        logits = hidden(hidden_1, activation, 'hidden_2', num_classes)

    return logits


def dnn_model(input_pl, activation=tf.nn.relu, dropout=None):
    """ DNN Model
    Parameters:
        input_pl: The input data placeholder.
        activation: The activation function.
        dropout: Whether use the dropout.
    Returns:
        logits: The model output value for each category.
     """
    flatten = tf.reshape(input_pl, [-1, width * height * num_channels])
    hidden_1 = hidden(flatten, activation, 'hidden_1', 512, dropout)
    hidden_2 = hidden(hidden_1, activation, 'hidden_2', 256, dropout)
    hidden_3 = hidden(hidden_2, activation, 'hidden_3', 128, dropout)
    logits = hidden(hidden_3, activation, 'hidden_4', 11)
    return logits


def eval_in_batches(data, sess, eval_prediction, eval_placeholder):
    """Get all predictions for a dataset by running it in small batches.
    Parameters:
        data: The evaluation data set.
        sess: The session with the graph.
        eval_prediction: The evaluation operator, which output the logits.
        eval_placeholder: The placeholder of evaluation data in the graph.
    Returns:
        predictions: The eval result of the input data, which has the format
                    of [size, num_classes]
    """
    size = data.shape[0]
    if size < batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, num_classes), dtype=np.float32)
    for begin in range(0, size, batch_size):
        end = begin + batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_placeholder: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_placeholder: data[-batch_size:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


def build_data(samples, labels):
    """ Build the train and test set.
    Parameters:
        samples: The whole samples we have.
        labels: The samples' labels correspondently.
    Returns:
        train_data: The train set data.
        train_labels: The train data's category labels.
        test_data: The test set data.
        test_labels: The test data's category labels.
    """
    num_samples = len(samples)
    indexes = list(range(num_samples))
    np.random.shuffle(indexes)
    num_train = int(train_ratio * num_samples)
    # Get the indexes of train data and test data.
    train_indexes = indexes[0:num_train]
    test_indexes = indexes[num_train:num_samples]

    # Build the train data and test data.
    train_data = samples[train_indexes]
    train_labels = labels[train_indexes]
    test_data = samples[test_indexes]
    test_labels = labels[test_indexes]
    print("size of train data ",train_data.shape,"size of test data:",test_data.shape)
    return train_data, test_data, \
           train_labels, test_labels, \
           train_indexes, test_indexes


def accuracy_snr(predictions, labels, indexes, snrs, samples_snr):
    """ Compute the error rate of difference snr.
    Parameters:
        predictions:
        labels:
        indexes:
        snrs:
        samples_snr:
    Returns:
        acc_snr
    """
    labels = labels.reshape([len(labels), ])
    predict_snr = samples_snr[indexes]

    acc_snr = dict()
    for snr in snrs:
        idx = (predict_snr == snr).reshape([len(labels)])
        samples_temp = predictions[idx]
        labels_temp = labels[idx]
        acc_snr[snr] = accuracy_compute(samples_temp, labels_temp)
    return acc_snr


def acc_snr_show(snrs, acc_snr, path):
    """ Show the train procedure.
    Parameters:
        sd
    Returns:
        Hello
    """
    # Plot accuracy curve
    plt.plot(snrs, list(map(lambda x: acc_snr[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN Classification Accuracy with Different SNR")
    plt.savefig(path)


def confusion_matrix(predict, labels, num_classes):
    """ Show the confusion of predict.
    Parameters:
        num_classes: The count of different classes.
        predict: The predict result of samples.
        labels: The real class of the samples.
    Returns:
        conf_norm: The normalized confusion matrix.
    """
    # Compute the count of correct and error samples in each snr.
    conf = np.zeros([num_classes, num_classes])
    for i in range(0, len(labels)):
        j = labels[i]
        k = np.argmax(predict[i])
        conf[j, k] = conf[j, k] + 1

    # Compute the count of correct and error ratio in each snr.
    # =====confusion matrix=====.
    conf_norm = np.zeros([num_classes, num_classes])
    for i in range(0, num_classes):
        conf_norm[i, :] = conf[i, :] / np.sum(conf[i, :])

    return conf_norm


def plot_confusion_matrix(conf_matrix, labels=[],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, name=None):
    """ Plot the confusion matrix.
    Parameter:
        conf_matrix:
        labels:
        title:
        cmap:
        name:
    Returns:
        None.
    """
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if name is None:
        plt.show()
    else:
        plt.savefig(name)


def main():
    # Define the input data.
    input_shape = [batch_size, height, width, num_channels]
    types=['DSB','USB','LSB','CPFSK','8PSKGray']
    # Load the train data and test data.
    # samples, labels, mod2cate, cate2mod, snrs, mods, samples_snr = \
    #     load_data('./data/RML2016.10a_dict.dat', input_shape)

    samples, labels, mod2cate, cate2mod=load_types(types)

    train_data, test_data, \
    train_labels, test_labels, \
    train_indexes, test_indexes = build_data(samples, labels)

    # Define the input placeholder.
    train_data_node = tf.placeholder(tf.float32, shape=[None, height, width, num_channels])
    train_labels_node = tf.placeholder(tf.int64, shape=[None])
    # eval_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, num_channels))

    # Model.
    # logits = cnn_2_model(train_data_node, lrelu, dropout)
    logits=classier_mlp(train_data_node,base=8,batch_Norm=True)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=logits))

    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classier')
    # Use simple adam for the optimization.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
    optimizer = layers.optimize_loss(loss=loss, learning_rate=learning_rate,
                           optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                             beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer,
                           variables=theta, global_step=global_step,
                           summaries=['gradient_norm'])
    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    correct_prediction = tf.equal(tf.argmax(train_prediction, 1), train_labels_node)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('value', accuracy)

    # saver = tf.train.Saver()

    merged = tf.summary.merge_all()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    # Create a local session to run the training.
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter('./log/', sess.graph)
        # train_writer=tf.train.SummaryWriter('./log/', sess.graph)

        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        print('Initialized!')

        # Loop through training steps.
        num_train = len(train_labels)
        max_step_train = int(num_epoch * num_train / batch_size)
        for step in range(max_step_train):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * batch_size) % (num_train - batch_size)
            batch_data = train_data[offset:(offset + batch_size), ...]
            batch_labels = train_labels[offset:(offset + batch_size)]

            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}

            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)

            # print some extra information once reach the evaluation frequency
            if step % train_show_step == 0:
                # fetch some extra nodes' data
                summary_str, loss_step, train_accu = \
                    sess.run([merged, loss, accuracy],
                             feed_dict=feed_dict)

                # eval_acc = accuracy(predictions, batch_labels, 'train_accuracy')
                print('Step: %d(epoch %.2f)  loss: %.3f, train_accuracy: %.3f%%' %
                      (step, float(step) * batch_size / num_train, loss_step, train_accu * 100))

                if step % test_show_step == 0:  # Test the test set.
                    test_predictions = eval_in_batches(test_data, sess, train_prediction, train_data_node)
                    print('Test accuracy: %.3f%% ' % accuracy_compute(test_predictions, test_labels))

                train_writer.add_summary(summary_str, step)
                # train_writer.flush()

        train_writer.close()

        # checkpoint_file = os.path.join(log_dir, 'model_final.ckpt')
        # saver.save(sess, checkpoint_file)

        test_predictions = eval_in_batches(test_data, sess, train_prediction, train_data_node)
        # acc_snr = accuracy_snr(test_predictions, test_labels,
        #                        test_indexes, snrs, samples_snr)
        # acc_snr_show(snrs, acc_snr, './log/accu_snrs.jpg')

        # Compute the confusion matrix.
        conf_matrix = confusion_matrix(test_predictions, test_labels, num_classes)

        # plot_confusion_matrix(conf_matrix, labels=mods, name='./log/conf_matrix.jpg')
        plot_confusion_matrix(conf_matrix, labels=types, name='./log/conf_matrix.jpg')



if __name__ == '__main__':
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    main()

