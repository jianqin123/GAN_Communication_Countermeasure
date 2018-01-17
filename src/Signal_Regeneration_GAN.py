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
import  math


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
interval_save=1000
#interval to save optimazaiton point
interval_points=100

# code version
model_name="model_WGAN"

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True





def buildData(type,number,result_save):
    path = os.path.join('..', 'signal_data',type)
    print("read real data from {} ".format(path))
    data_real = np.ones((1, number))
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.mat':
            if type == '8PSKGray':
                signal = sio.loadmat(os.path.join(path, file))['x']
            else:
                signal = sio.loadmat(os.path.join(path, file))['s']

            signal = np.reshape(signal[:, 0:-(signal.shape[1] % number)], [-1, number])


            if type in set(['8PSKGray','CPFSK']):
                data_I =  signal.real
            else:
                data_I=signal
            print("shape of data_I",data_I.shape)

            data_real = np.vstack((data_real, data_I))

            draw_Series_Grid(data_I, "real_data", result_save, "real_data_{}".format(os.path.splitext(file)[0]), row=2,
                             col=2)
    data_real = data_real[1:]
    return data_real
#lrelu activation  function
def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class model :
    def __init__(self, params):
        self.batch_size=params.batch_size
        self.is_mlp_g=params.is_mlp_g
        self.is_mlp_c=params.is_mlp_c
        self.max_iter_step=params.max_iter_step
        self.Noise=params.Noise
        self.loss=params.loss
        self.type=params.type
        self.number=params.number

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
    def get_Signal_From(self,mat):
        len=mat['s'].shape[1]
        data = np.reshape(mat['s'][0][0:len - len % self.number].copy(), (-1, self.number))
        indexs=np.random.randint(0,data.shape[0],self.batch_size)
        return data[indexs,:]

    def get_Signal_FromData(self,data_real):
        indexs = random.sample(range(data_real.shape[0]), self.batch_size)
        return data_real[indexs, :]


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


    def generator_mlp(self,train,base=8,recover=False):
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        if self.Noise & (not recover):
            noise0 = noise_dist.sample(train.get_shape().as_list())
            train = train + noise0
        len=int(math.log(int(self.number/base),2))
        size=[ base * pow(2,i) for i in range(len)]
        print('generator_mlp.size ',size)
        train_names=[train.name]
        for i in range(len) :
            train= ly.fully_connected(
               train, size[i], activation_fn=None, normalizer_fn=ly.batch_norm,scope='g{}'.format(i+1))
            print("train_{}.name".format(i+1),train.name)
            train_names.append(train.name)
            if  self.Noise & (not recover):
                noise = noise_dist.sample(train.get_shape().as_list())
                train = train + noise

        train1 = ly.fully_connected(
            train, self.number, activation_fn=None, normalizer_fn=ly.batch_norm,scope='g')

        train1= tf.reshape(train1, tf.stack([self.batch_size, 1, self.number, 1]),name="train")
        train_names.append(train1.name)
        return train1,train_names

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

    def critic_mlp(self,img,base=16,reuse=False):
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        len=int(math.log(int(self.number/base),2))
        size=[int(self.number/pow(2,i+1)) for i in range(len)]
        # size = [base * pow(2, i) for i in range(len)]
        img=tf.reshape(img, [self.batch_size, -1])

        if self.Noise:
            noise0 = noise_dist.sample(img.get_shape().as_list())
            img = img + noise0

        with tf.variable_scope('critic') as scope:
            if reuse:
                scope.reuse_variables()
            for i in range(len):
                img = ly.fully_connected(img, size[i], activation_fn=tf.nn.relu)
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
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        z = noise_dist.sample(( self.batch_size, int(3*self.number/4)))
        recover=tf.Variable(False,"recover")
        generator = self.generator_mlp if self.is_mlp_g else self.generator_conv
        critic =  self.critic_mlp if  self.is_mlp_c else  self.critic_conv
        with tf.variable_scope('generator'):
            train,train_names = generator(z,recover=recover)
        real_data = tf.placeholder(
            dtype=tf.float32, shape=( self.batch_size, 1, self.number, 1))
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

        return opt_g, opt_c, real_data,train_names


    def train(self):

        tf.reset_default_graph()
        with tf.device(device):
            opt_g, opt_c, real_data,train_names = self.build_graph()

        label="G_{}_D_{}_{}_{}_{}".format("M" if self.is_mlp_g else "C","M" if self.is_mlp_c else "C" ,self.loss,self.number,"Noise" if self.Noise else "")
        print(label+"  " + str(train_names))
       #模型保存路径和结果保存路径
        model_save=os.path.join('..','model',self.type,label)
        result_save=os.path.join('..','result',self.type,label)
        if not os.path.exists(model_save):
            os.makedirs(model_save)

        if not os.path.exists(result_save):
            os.makedirs(result_save)

        data_real=buildData(self.type,self.number)

        print("amount of real data to train :",data_real.shape)

        saver = tf.train.Saver(max_to_keep=1+int(self.max_iter_step/interval_save))

        def next_feed_dict():
            train_sig=np.reshape(self.get_Signal_FromData(data_real),(self.batch_size,1,self.number,1))
            feed_dict = {real_data: train_sig}
            return feed_dict
        data_optima=[]

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.max_iter_step):
                #save model
                if(i%interval_save==0):
                    saver.save(sess,os.path.join(model_save,model_name),global_step=i)

                if  (i < 25 or i % 500 == 0):
                    citers = 100
                else:
                    citers = Citers

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
        np.save(result_save,data_optima)
        x=data_optima[:,0]
        pl1,=pl.plot(x,data_optima[:,1],"r")
        pl2,=pl.plot(x,data_optima[:,2],"g")
        pl.xlabel("iteration_counts")
        pl.ylabel("loss")
        pl.title("optimazation process for {}".format(label))
        pl.legend(handles=[pl1,pl2],labels=['opt_c_value_mean','opt_g_value'])
        pl.savefig(os.path.join(result_save,"pic_optima"))
        sess.close()
        return  train_names,self.type,label,self.number,model_save,result_save

def oneSolution(batch_size,is_mlp_g,is_mlp_c,max_iter_step,Noise,loss,type,number):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=batch_size,
                        help='the size of one epoch')
    parser.add_argument('--is-mlp-g', type=bool, default=is_mlp_g)
    parser.add_argument('--is-mlp-c', type=bool, default=is_mlp_c)
    parser.add_argument('--max-iter-step',type=int,default=max_iter_step)
    parser.add_argument('--Noise', type=bool, default=Noise)
    parser.add_argument('--loss', type=str, default=loss)
    parser.add_argument('--type', type=str, default=type)
    parser.add_argument('--number', type=int, default=number)
    return model(parser.parse_args()).train()



def searchParam():    # main()
    # batch_size_=[16,32,64,128,256]
    real_data_ = ['AM', 'DSB', 'USB', 'LSB', 'FM']
    # real_data_=['AM','DSB']
    number_=[512,1024]
    for real in real_data_:
        for num in number_:
            number=num
            real_data=real
            loss='WGAN'
            is_mlp_g=True
            batch_size=128
            is_mlp_c=True
            max_iter_step=5001
            Noise=True
            train_names,type, label,number, model_save, result_save=oneSolution(batch_size,is_mlp_g,is_mlp_c,max_iter_step,Noise,loss,real_data,number)
            #恢复数据`
            recover_model(model_save,train_names,result_save,type, label,model_interval=interval_save,model_number=1+int(max_iter_step/interval_save))



def draw_Series_Grid(seq,title,save_to,name,row,col):
   fig ,axes=plt.subplots(row,col,sharex=True,sharey=True)
   if not os.path.exists(save_to):
       os.makedirs(save_to)
   for i in range(row):
       for j in range(col):
           axes[i,j].plot(range(seq.shape[1]),seq[i+j][:],'.-')
   fig.suptitle(title)
   plt.savefig(os.path.join(save_to,name))
   fig.clear()

def recover_pic(model_save,result_save,type,label,tensorname,pic_number=2, iterations = 16000):
    print("regenerate data at {} {} {} ".format(type,label,iterations))

    model_path=os.path.join(model_save,"{}-{}".format('model_WGAN',iterations))
    saver = tf.train.import_meta_graph(model_path+ ".meta")
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)
        recover=tf.get_default_graph().get_tensor_by_name("Variable:0")
        dict={recover:True}

        print("restore model from {} to regenerate data {}".format(model_path,str(tensorname)))
        train_names=[tf.get_default_graph().get_tensor_by_name(tensorname[i]) for i in range(len(tensorname))]
        train_data_all = sess.run(train_names,dict)
        indexs=random.sample(range(train_data_all[0].shape[0]),pic_number*pic_number)
        for i in range(len(tensorname)):
            train = np.array(train_data_all[i]).reshape((train_data_all[i].shape[0],-1))
            title=" {}_signal on model_{} layer_{} iter_{}".format(type,label,i,iterations)
            draw_Series_Grid(train[indexs],title,result_save,name="signal_{}_layer_{}".format(iterations,i),row=pic_number,col=pic_number)


def recover_model(model_save,train_names,result_save,type,label,model_number = 14,model_interval = 3000,
                  pic_number=2):
    for i in range(model_number):
        recover_pic(model_save,result_save,type,label, iterations=i* model_interval,pic_number=pic_number,tensorname=train_names)

searchParam()
