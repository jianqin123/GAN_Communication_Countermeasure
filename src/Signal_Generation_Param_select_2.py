from __future__ import print_function
import os
import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from six.moves import xrange
import tensorflow.contrib.slim as slim

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as pl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import os
import  scipy.io as sio
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
# from load_svhn import load_svhn
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import  struct
import  pickle
import  random
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import  random


learning_rate_ger = 5e-5
learning_rate_dis = 5e-5
device = '/gpu:0'

# update Citers times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)
Citers = 5
# the upper bound and lower bound of parameters in critic
clamp_lower = -0.01
clamp_upper = 0.01
# whether to use mlp or dcgan stucture

# whether to use adam for parameter update, if the flag is set False, use tf.train.RMSPropOptimizer
# as recommended in paper
is_adam = False
# if 'gp' is chosen the corresponding lambda must be filled
lam = 10.
# max iter step, note the one step indicates that a Citers updates of critic and one update of generator
#interval to save model
interval_save=2000
#interval to save optimazaiton point
interval_points=100
#signal type
type="AM"
# code version
#number of data point
number = 32 * 8
z_dim=128
number=32*8

#path of real data
path = os.path.join("..", os.path.join(os.path.join("signal_data", "AM"), "am1"))
model_name="model_WGAN"

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

config.gpu_options.per_process_gpu_memory_fraction = 0.8


#lrelu activation  function
def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class model :
    def __init__(self, params):
        self.batch_size=params.batch_size
        self.z_dim=params.z_dim
        self.is_mlp_g=params.is_mlp_g
        self.is_mlp_c=params.is_mlp_c
        self.max_iter_step=params.max_iter_step
        self.Noise=params.Noise
        self.loss=params.loss

# read real data
    def get_signal(self,Xd, signal):
        data = []
        for key in Xd.keys():
            if (key[0] == signal):
                data.append(Xd[key])
        data = np.array(data).reshape((-1, 16, 16,1))

        # regulization

        min = 5 / 4 * data.min()
        max = 11 / 10 * (data.max() - min)
        data = 1 / (max) * (data - min)

        indexs = random.sample(list(range(0, data.shape[0])), self.batch_size)
        samples = data[indexs, :]
        return samples

    #get real data from mat file
    def get_Signal_From(self,mat,number):
        len=mat['s'].shape[1]
        data = np.reshape(mat['s'][0][0:len - len % number].copy(), (-1, number))
        indexs=np.random.randint(0,data.shape[0],self.batch_size)
        return data[indexs,:]



    #相对于原始的generator_cov 的二维卷积核 该为一维卷积核
    def generator_conv(self,z,recover=False):
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        if self.Noise & (not recover):
            noise0 = noise_dist.sample(z.get_shape().as_list())
            z = z + noise0

        train1 = ly.fully_connected(
            z, 4 * 4 * 32, activation_fn=lrelu, normalizer_fn=ly.batch_norm,scope="g1")

        train1 = tf.reshape(train1, (-1, 1, 16, 32),name="train")
        if self.Noise & (not recover):
            noise1=noise_dist.sample(train1.get_shape().as_list())
            train1=train1+noise1
        # 3*3 kernel
        train2 = ly.conv2d_transpose(train1, 256, (1,2), stride=(1,2),
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='valid',
                                    weights_initializer=tf.random_normal_initializer(0, 0.02),scope="g2")
        if self.Noise & (not recover):
            noise2=noise_dist.sample(train2.get_shape().as_list())
            train2=train2+noise2
        train3 = ly.conv2d_transpose(train2, 128, (1,2), stride=(1,2),
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='valid',
                                    weights_initializer=tf.random_normal_initializer(0, 0.02),scope="g3")
        if self.Noise & (not recover):
            noise3=noise_dist.sample(train3.get_shape().as_list())
            train3=train3+noise3
        train4 = ly.conv2d_transpose(train3, 64,(1,2), stride=(1,2),
                                     activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='valid',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02),scope="g4")
        if self.Noise & (not recover):
            noise4 = noise_dist.sample(train4.get_shape().as_list())
            train4= train4 + noise4

        train5 = ly.conv2d_transpose(train4, 1, (1,2), stride=(1,2),
                                     activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='valid',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02), scope="g5")
        return train5


    def generator_mlp(self,z,size=[32,64,128],recover=False):
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        if self.Noise & (not recover):
            noise0 = noise_dist.sample(z.get_shape().as_list())
            z = z + noise0

        train1= ly.fully_connected(
            z, size[0], activation_fn=None, normalizer_fn=ly.batch_norm,scope='g1')
        if  self.Noise & (not recover):
            noise1 = noise_dist.sample(train1.get_shape().as_list())
            train1 = train1 + noise1

        train2 = ly.fully_connected(
            train1, size[1], activation_fn=None, normalizer_fn=ly.batch_norm,scope='g2')

        if  self.Noise & (not recover):
            noise2= noise_dist.sample(train2.get_shape().as_list())
            train2= train2 + noise2

        train3 = ly.fully_connected(
            train2, size[2], activation_fn=None, normalizer_fn=ly.batch_norm,scope='g3')
        if  self.Noise & (not recover):
            noise3 = noise_dist.sample(train3.get_shape().as_list())
            train3 = train3+ noise3

        train4 = ly.fully_connected(
            train3, number, activation_fn=None, normalizer_fn=ly.batch_norm,scope='g4')


        train4= tf.reshape(train4, tf.stack([self.batch_size, 1, number, 1]),name="train")
        return train4


    #shape of  image is (1,256)
    def critic_conv(self,img, reuse=False):
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        if self.Noise:
            noise0 = noise_dist.sample(img.get_shape().as_list())
            img=img+noise0
        with tf.variable_scope('critic') as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            img = ly.conv2d(img, num_outputs=size*8, kernel_size=(1,16),
                            stride=2, activation_fn=lrelu)
            img = ly.conv2d(img, num_outputs=size *6, kernel_size=(1,8),
                            stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
            img = ly.conv2d(img, num_outputs=size * 4, kernel_size=(1,4),
                            stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
            img = ly.conv2d(img, num_outputs=size * 2, kernel_size=(1,2),
                            stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)


            logit = ly.fully_connected(tf.reshape(
                img, [self.batch_size, -1]), 1, activation_fn=None)
        return logit

    def critic_mlp(self,img, size=[128,64,32],reuse=False):
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        if self.Noise:
            noise0 = noise_dist.sample(img.get_shape().as_list())
            img = img + noise0
        with tf.variable_scope('critic') as scope:
            if reuse:
                scope.reuse_variables()
            img = ly.fully_connected(tf.reshape(
                img, [self.batch_size, -1]), size[0], activation_fn=tf.nn.relu)
            img = ly.fully_connected(img, size[1],
                                     activation_fn=tf.nn.relu)
            img = ly.fully_connected(img, size[2],
                                     activation_fn=tf.nn.relu)
            logit = ly.fully_connected(img, 1, activation_fn=None)
        return logit

    #cal loss of LS_GAN
    def ls_GAN(self,true_logit,fake_logit):
        c_loss = 0.5 * (tf.reduce_mean((true_logit - 1) ** 2) + tf.reduce_mean(fake_logit ** 2))
        g_loss = 0.5 * tf.reduce_mean((fake_logit - 1) ** 2)
        return c_loss,g_loss


    #cal loss of WGAN
    def wgan_loss(self,real_data,train,critic,true_logit,fake_logit):
        c_loss = tf.reduce_mean(fake_logit - true_logit)
        alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
        alpha = alpha_dist.sample(( self.batch_size, 1, 1, 1))
        interpolated = real_data + alpha * (train - real_data)
        inte_logit = critic(interpolated, reuse=True)

        gradients = tf.gradients(inte_logit, [interpolated, ])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)

        c_loss += lam * gradient_penalty

        g_loss = tf.reduce_mean(-fake_logit)
        return c_loss, g_loss






    def build_graph(self):
        #     z = tf.placeholder(tf.float32, shape=(batch_size, z_dim))
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        z = noise_dist.sample(( self.batch_size, self.z_dim))
        recover=tf.Variable(False,"recover")
        generator = self.generator_mlp if self.is_mlp_g else self.generator_conv
        critic =  self.critic_mlp if  self.is_mlp_c else  self.critic_conv
        with tf.variable_scope('generator'):
            train = generator(z,recover=recover)
        train_name=train.name
        real_data = tf.placeholder(
            dtype=tf.float32, shape=( self.batch_size, 1, number, 1))
        true_logit = critic(real_data)
        fake_logit = critic(train, reuse=True)
        if self.loss=='WGAN':
            c_loss, g_loss=self.wgan_loss(real_data,train,critic,true_logit,fake_logit)
        if self.loss=="LSGAN":
            c_loss, g_loss = self.ls_GAN( true_logit, fake_logit)

        theta_g = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        theta_c = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
                                 optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                                   beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer,
                                 variables=theta_g, global_step=counter_g,
                                 summaries=['gradient_norm'])
        counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_c = ly.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
                                 optimizer=partial(tf.train.AdamOptimizer, beta1=0.5,
                                                   beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer,
                                 variables=theta_c, global_step=counter_c,
                                 summaries=['gradient_norm'])

        return opt_g, opt_c, real_data,train_name


    def train(self):

        tf.reset_default_graph()
        with tf.device(device):
            opt_g, opt_c, real_data,train_name = self.build_graph()

        label="G_{}_D_{}_{}_{}".format("M" if self.is_mlp_g else "C","M" if self.is_mlp_c else "C" ,self.loss,"Noise" if self.Noise else "")
        print(label+"  " + train_name)
        save(os.path.join('..','save',type,'log.txt'),contents=label)

        save_path = os.path.join(os.path.join(os.path.join("..", "save"), type), label)
        save_opt = os.path.join(save_path, "data_optimazation")

        saver = tf.train.Saver(max_to_keep=int(self.max_iter_step/interval_save))

        # path = "D:\\gan_intro\\WGAN-tensorflow\\MNIST_data\\t10k-images.idx3-ubyte"  # the mnist data

        # path =os.path.join(os.path.join("..", "signal_data"),"RML2016.10a_dict.dat")
        # print("read real data from {} ".format(path))
        # Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

        print("read real data from {} ".format(path))
        mat = sio.loadmat(path)


        if not os.path.exists(save_path):
            os.makedirs(save_path)

        def next_feed_dict():
            # train_sig = get_signal(Xd, "BPSK", batch_size)
            train_sig=np.reshape(self.get_Signal_From(mat,number),(self.batch_size,1,number,1))
            feed_dict = {real_data: train_sig}
            return feed_dict
        data_optima=[]

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.max_iter_step):
                #save model
                if(i%interval_save==0):
                    saver.save(sess,os.path.join(save_path,model_name),global_step=i)

                if  (i < 25 or i % 500 == 0):
                    citers = 100
                else:
                    citers = Citers

                opt_c_value_mean = np.zeros(self.batch_size, np.float16)
                opt_c_value_mean = 0
                for j in range(citers):
                    feed_dict = next_feed_dict()
                    opt_c_value = sess.run([opt_c], feed_dict=feed_dict)
                    opt_c_value_mean += opt_c_value[0]
                opt_c_value_mean /= citers
                feed_dict = next_feed_dict()
                opt_g_value = sess.run([opt_g], feed_dict=feed_dict)
                if(i%interval_points==0):
                    data_optima.append([i,opt_c_value_mean,opt_g_value[0]])
                    print(" {} / {}   opt_c_value_mean:{}   opt_g_value:{} ".format(i, self.max_iter_step, opt_c_value_mean,
                                                                           opt_g_value))
        # save  the the optimzation data when training this model
        data_optima=np.array(data_optima)
        np.save(save_opt,data_optima)
        x=data_optima[:,0]
        pl1,=pl.plot(x,data_optima[:,1],"r")
        pl2,=pl.plot(x,data_optima[:,2],"g")
        pl.xlabel("iteration_counts")
        pl.ylabel("loss")
        pl.title("optimazation process for {}".format(label))
        pl.legend(handles=[pl1,pl2],labels=['opt_c_value_mean','opt_g_value'])
        pl.savefig(os.path.join(save_path,"pic_optima"))
        sess.close()
        return  train_name,label

def oneSolution(batch_size,z_dim,is_mlp_g,is_mlp_c,max_iter_step,Noise,loss):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=batch_size,
                        help='the size of one epoch')
    parser.add_argument('--z-dim',type=int,default=z_dim)
    parser.add_argument('--is-mlp-g', type=bool, default=is_mlp_g)
    parser.add_argument('--is-mlp-c', type=bool, default=is_mlp_c)
    parser.add_argument('--max-iter-step',type=int,default=max_iter_step)
    parser.add_argument('--Noise', type=bool, default=Noise)
    parser.add_argument('--loss', type=str, default=loss)
    return model(parser.parse_args()).train()



def searchParam():    # main()
    # batch_size_=[16,32,64,128,256]
    batch_size_=[128]
    z_dim_=[128]
    loss_=['WGAN','LSGAN']
    is_mlp_c_=[True]
    is_mlp_g_=[True,False]
    Noise_=[True,False]
    max_iter_step_=[20001]

    for l in loss_:
        loss=l
        for z in z_dim_:
            z_dim = z
            for c in is_mlp_c_:
                is_mlp_c = c
                for g in is_mlp_g_:
                    is_mlp_g=g
                    for max in max_iter_step_:
                        max_iter_step = max
                        for batch_ in batch_size_:
                            for no in Noise_:
                                Noise=no
                                batch_size=batch_
                                train_name,label=oneSolution(batch_size,z_dim,is_mlp_g,is_mlp_c,max_iter_step,Noise,loss)
                                #恢复数据
                                recover_model(label,model_interval=interval_save,model_number=int(max_iter_step/interval_save),train_name=train_name)



def draw_Series_Grid(seq,title,save_to,name,row,col,N=64):
   fig ,axes=plt.subplots(row,col,sharex=True,sharey=True)
   if not os.path.exists(save_to):
       os.makedirs(save_to)
   for i in range(row):
       for j in range(col):
           index=random.sample(range(seq.shape[0]),1)[0]
           print("get data at index {}".format(index))
           axes[i,j].plot(range(N),seq[index][:N],'.-')
   fig.suptitle(title)
   print('save to {}'.format(save_to))
   plt.savefig(os.path.join(save_to,name))
   fig.clear()

def recover_pic(label,type="AM",pic_number=2, iterations = 16000,tensorname="generator/train:0",seq_number=64):
    print("regenerate data at {} {} {} ".format(type,label,iterations))
    path = os.path.join("..","save",type,label,"{}-{}".format("model_WGAN",iterations))
    save_to =os.path.join("..","result",type,label)
    saver = tf.train.import_meta_graph(path + ".meta")
    print("restore model from {}".format(path))
    with tf.Session(config=config) as sess:
        saver.restore(sess, path)
        recover=tf.get_default_graph().get_tensor_by_name("Variable:0")
        dict={recover:True}
        train = sess.run(tf.get_default_graph().get_tensor_by_name(tensorname),dict)
        train = np.array(train).reshape((-1, number))
        draw_Series_Grid(train," {}_signal on model_{} iter_{}".format(type,label,iterations),
                         save_to,name="signal_{}".format(iterations),row=pic_number,col=pic_number,N=seq_number)


def recover_model(label,model_number = 14,model_interval = 3000,
                  pic_number=4,seq_number = 64 ,train_name="generator/g4/Tanh:0"):
    for i in range(model_number):
        recover_pic(label, iterations=(i +1)* model_interval,pic_number=pic_number,tensorname=train_name,seq_number=seq_number)
searchParam()
