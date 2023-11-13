#!/usr/bin/env python
# coding: utf-8

# # VAE model
# ##### Script for training model


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, decomposition, manifold, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from tensorflow.keras import layers
import pickle
import time
from numpy import load
from matplotlib import pyplot
import pickle
import argparse 

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

from tensorflow.compat.v1.keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D,  Concatenate
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Dense, Reshape, Flatten
from tensorflow.compat.v1.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import mse, binary_crossentropy, logcosh
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.stats import wasserstein_distance
import pandas as pd
from utils import sum_channels_parallel_ as sum_channels_parallel
from sklearn.metrics import mean_absolute_error


def save_scales(model_name, scaler_means, scaler_scales):
    out_fnm = f"{model_name}_scales.txt"
    res = "#means"
    for mean_ in scaler_means:
        res += "\n" + str(mean_)
    res += "\n\n#scales"
    for scale_ in scaler_scales:
        res += "\n" + str(scale_)
    with open(f"./models/{out_fnm}", mode="w") as f:
        f.write(res)


def main(name,epochs,data_dir):

    NAME=name
    os.makedirs("images",exist_ok=True)
    os.makedirs("models",exist_ok=True)
    
    # load the dataset
    data = load(os.path.join(data_dir,'data_nonrandom_responses.npz'))["arr_0"]
    print('Loaded: ',  data.shape, "max:", data.max())

    data_cond = load(os.path.join(data_dir,'data_nonrandom_particles.npz'))["arr_0"]
    data_cond = pd.DataFrame(data_cond, columns =['Energy','Vx','Vy',	'Vz',	'Px',	'Py',	'Pz',	'mass',	'charge'])
    COND_DIM=data_cond.shape[-1]
    print('Loaded cond: ',  data_cond.shape)

    
    #preprocess data
    data = np.log(data+1)
    data = np.float32(data)
    print("data max", data.max(), "min", data.min())
    data_cond = np.float32(data_cond)

    
    #train/test split
    x_train, x_test, y_train, y_test, = train_test_split(data, data_cond, test_size=0.2, shuffle=False)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    #scale cond data
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    print("cond max", y_train.max(), "min", y_train.min())
    
    #save scales
    save_scales(NAME, scaler.mean_, scaler.scale_)


    # make tf datasets
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size=128)
    dataset_cond = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size=128)
    dataset_with_cond = tf.data.Dataset.zip((dataset,dataset_cond)).shuffle(12800)

    val_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size=128)
    val_dataset_cond = tf.data.Dataset.from_tensor_slices(y_test).batch(batch_size=128)
    val_dataset_with_cond = tf.data.Dataset.zip((val_dataset,val_dataset_cond)).shuffle(12800)


    ############################ Define Models ############################
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    latent_dim = 10
    cond_dim = 9
    poz_dim = 6


    ############################ encoder ############################

    input_img = Input(shape=[56,30,1],name='input_img')
    input_cond = Input(shape=cond_dim,name='input_cond')
    x = Conv2D(32, kernel_size=4, strides=2, padding='same')(input_img)
    x = Conv2D(64, kernel_size=4, strides=2,padding='same')(x)
    x = Conv2D(128, kernel_size=4, strides=2,padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    x = layers.concatenate([input_cond,x])
    x = layers.Dense(latent_dim*2, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([input_img, input_cond], [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    ############################ decoder ############################


    x = Input(shape=(latent_dim,))
    cond = Input(shape=(cond_dim,))
    inputs = Concatenate(axis=1)([x, cond])

    g = Dense(7*4*128)(inputs)
    g = Reshape((7,4,128))(g)

    g = UpSampling2D()(g)
    g = Conv2D(128, kernel_size=4, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0)(g)

    g = UpSampling2D()(g)
    g = Conv2D(64, kernel_size=4, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0)(g)

    g = UpSampling2D()(g)
    g = Conv2D(32, kernel_size=4, padding='same')(g)
    g = BatchNormalization()(g)
    g = LeakyReLU(alpha=0)(g)

    outputs = Conv2D(1, kernel_size=(1,3) ,activation='relu')(g)

    generator = Model([x, cond], outputs, name='generator')
    generator.summary()


    # define losses
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    d_acc_r = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    d_acc_f = keras.metrics.BinaryAccuracy(name="d_acc_r", threshold=0.5)
    g_acc = keras.metrics.BinaryAccuracy(name="g_acc_g", threshold=0.5)

    # define optimizer
    vae_optimizer = tf.keras.optimizers.RMSprop(1e-4)


    #trainin params

    EPOCHS = epochs
    noise_dim = 10
    num_examples_to_generate = 16

    # Seed to reuse for generating samples for comparison during training
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    seed_cond = y_test[:num_examples_to_generate]


    ### function to calculate ws distance between orginal and generated channels
    org=np.exp(x_test)-1
    ch_org = np.array(org).reshape(-1,56,30)
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values 
    del org


    def calculate_ws_ch(n_calc):
      ws= [0,0,0,0,0]
      for j in range(n_calc):
        z = np.random.normal(0,1,(x_test.shape[0],10))
        z_c = y_test
        results = generator.predict([z,z_c])
        results = np.exp(results)-1
        try:
          ch_gen = np.array(results).reshape(-1,56,30)
          ch_gen = pd.DataFrame(sum_channels_parallel(ch_gen)).values
          for i in range(5):
            ws[i] = ws[i] + wasserstein_distance(ch_org[:,i], ch_gen[:,i])
          ws =np.array(ws)
        except ValueError as e:
          print(e)
      ws = ws/n_calc
      print("ws mean",f'{ws.sum()/5:.2f}', end=" ")
      for n,score in enumerate(ws):
        print("ch"+str(n+1),f'{score:.2f}',end=" ")



    ####################### training ##############################
    @tf.function
    def train_step(batch,step):

        images, cond = batch
        step=step
        BATCH_SIZE = tf.shape(images)[0]

        #train vae
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder([images, cond])
            reconstruction = generator([z, cond])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(tf.reshape(images,(-1,56,30,1)), reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = 0.7 * kl_loss + reconstruction_loss    
        grads = tape.gradient(total_loss, generator.trainable_weights+encoder.trainable_weights)
        vae_optimizer.apply_gradients(zip(grads, generator.trainable_weights+encoder.trainable_weights))       

        return total_loss, reconstruction_loss, kl_loss


    history = []
    def train(dataset, epochs):
      tf_step =tf.Variable(0, dtype=float)
      step=0
      generate_and_save_images(generator,
                               epochs,
                               [seed, seed_cond])
      #print(calculate_ws_mae(3))
      for epoch in range(epochs):
        start = time.time()

        for batch in dataset:
          total_loss, reconstruction_loss, kl_loss =train_step(batch,tf_step)
          history.append([total_loss, reconstruction_loss, kl_loss])
          tf_step.assign_add(1)
          step = step+1

          if step%100==0:
            print("%d [Total loss: %.2f] [Recon_loss: %.2f] [KL loss: %.2f]"% (
                step, total_loss, reconstruction_loss, kl_loss))

          if step%1000==0:
              generate_and_save_images(generator,
                               epochs,
                               [seed, seed_cond])

        generate_and_save_images(generator,
                                 epoch + 1,
                                 [seed, seed_cond]
                                 )

        # Save the model every epoch
        encoder.save_weights("./models/enc_"+NAME + "_"+ str(epoch) +".h5")
        generator.save_weights("./models/gen_"+NAME + "_"+ str(epoch) +".h5")
        np.savez("./models/history_"+NAME+".npz",np.array(history))

        calculate_ws_ch(min(epoch//5+1,5))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      return history



    def generate_and_save_images(model, epoch, test_input):
      # Notice `training` is set to False.
      # This is so all layers run in inference mode (batchnorm).
      predictions = model(test_input, training=False)

      fig, axs = plt.subplots(2, 7, figsize=(15,4))
      for i in range(0,14):
        if i <7:
            x = x_test[20+i].reshape(56,30)
        else:
            x = predictions[i-7].numpy().reshape(56,30)
        #x[x<=0]=x.max()*-0.1
        im = axs[i//7,i%7].imshow(x,interpolation='none', cmap='gnuplot')
        axs[i//7,i%7].axis('off')
        fig.colorbar(im, ax=axs[i//7,i%7])
      plt.show()
      plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))


    # ### Train model
    history=train(dataset_with_cond, EPOCHS)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'train VAE')
    parser.add_argument('--name', help='model name', required=True)
    parser.add_argument('--epochs',type=int, help='epochs to train', default=100)
    parser.add_argument('--data', help='path to data dir with data_nonrandom_responses.npz and data_nonrandom_particles.npz', default="data")
    args = parser.parse_args()

    main(args.name, args.epochs, args.data)

