from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.optimizers import RMSprop, Adam
from keras.layers.normalization import BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import model_from_json
# from keras.callbacks import Callback

from dotenv import load_dotenv, find_dotenv

from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

import os
import sys
import glob
import h5py
import numpy as np
# import matplotlib.pylab as plt
# import matplotlib.gridspec as gridspec
import time
import shutil
# Utils
sys.path.append("../utils")
import batch_utils
import general_utils


def SFCNN(nb_classes):
    """
    Build Convolution Neural Network

    args : nb_classes (int) number of classes

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(1, 160, 160)))
    model.add(Convolution2D(8, 3, 3, activation='relu', init="he_normal"))
    # model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', init="he_normal"))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', init="he_normal"))
    # model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', init="he_normal"))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', init="he_normal"))
    # model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', init="he_normal"))
    # model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', init="he_normal"))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', init="he_normal"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', init="he_normal"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', init="he_normal"))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu', init="he_normal"))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu', init="he_normal"))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu', init="he_normal"))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.name = "SFCNN"

    return model


def train(model, **kwargs):

    # Roll out the parameters
    nb_classes = kwargs["nb_classes"]
    num_frames = kwargs["num_frames"]
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    prob = kwargs["prob"]
    do_plot = kwargs["do_plot"]
    data_file = kwargs["data_file"]

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Load env variables
    model_dir = os.path.expanduser(os.environ.get("MODEL_DIR"))

    # Output path where we store experiment log and weights
    model_dir = os.path.join(model_dir, model.name)
    # Create if it does not exist
    general_utils.create_dir(model_dir)
    # Automatically determine experiment name
    list_exp = glob.glob(model_dir + "/*")
    # Create the experiment dir and weights dir
    exp_dir = os.path.join(model_dir, "Experiment_%s" % len(list_exp))
    weights_dir = os.path.join(exp_dir, "Weights")
    general_utils.create_dir([exp_dir, weights_dir])

    # Load test data in memory for fast error evaluation
    with h5py.File(data_file, "r") as hf:
        X_test, y_test = hf["test_data"][:5000, :, :, :], hf["test_label"][:5000]
        X_test = X_test / 255. - 0.5
        y_test_bin = np_utils.to_categorical(y_test, nb_classes=nb_classes)

    # Compile model.
    # opt = RMSprop(lr=5E-6, rho=0.9, epsilon=1e-06)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    general_utils.pretty_print("Compiling...")
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    # Batch generator
    DataAug = batch_utils.AugDataGenerator(data_file,
                                           batch_size=batch_size,
                                           prob=prob,
                                           dset="train",
                                           maxproc=4,
                                           num_cached=60,
                                           random_augm=False)
    DataAug.add_transform("h_flip")
    # DataAug.add_transform("v_flip")
    # DataAug.add_transform("fixed_rot", angle=40)
    DataAug.add_transform("random_rot", angle=40)
    # # DataAug.add_transform("fixed_tr", tr_x=40, tr_y=40)
    # DataAug.add_transform("random_tr", tr_x=40, tr_y=40)
    # DataAug.add_transform("fixed_blur", kernel_size=5)
    DataAug.add_transform("random_blur", kernel_size=7)
    # DataAug.add_transform("fixed_erode", kernel_size=4)
    DataAug.add_transform("random_erode", kernel_size=3)
    # # DataAug.add_transform("fixed_dilate", kernel_size=4)
    DataAug.add_transform("random_dilate", kernel_size=3)
    # # DataAug.add_transform("fixed_crop", pos_x=10, pos_y=10,
    #                          # crop_size_x=200, crop_size_y=200)
    DataAug.add_transform("random_crop", min_crop_size=140, max_crop_size=160)

    epoch_size = n_batch_per_epoch * batch_size

    json_string = model.to_json()
    with open(os.path.join(exp_dir, '%s_archi.json' % model.name), 'w') as f:
        f.write(json_string)

    # Save losses
    list_train_loss = []
    list_test_loss = []

    try:
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            # Start Epoch
            l_train_loss = []
            start = time.time()

            model.save_weights(os.path.join(weights_dir,
                                            '%s_weights_epoch%s.h5' %
                                            (model.name, e)),
                               overwrite=True)

            for X, y in DataAug.gen_batch():
                if do_plot:
                    general_utils.plot_batch(X, y, batch_size)
                # Convert y to binary matrix
                y = np_utils.to_categorical(y, nb_classes=2)
                train_loss = model.train_on_batch(X, y)
                l_train_loss.append(train_loss)
                batch_counter += 1
                progbar.add(batch_size, values=[("train loss", train_loss)])
                if batch_counter >= n_batch_per_epoch:
                    break
            print
            print 'Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start)
            y_test_pred = model.predict(X_test, verbose=0)
            train_loss = float(np.mean(l_train_loss))  # use float to make json serializable
            test_auc = roc_auc_score(y_test_bin, y_test_pred)
            test_loss = log_loss(y_test_bin, y_test_pred)
            print "Train loss:", train_loss, "Test loss:", test_loss, "Test AUC:", test_auc
            list_train_loss.append(train_loss)
            list_test_loss.append(test_loss)

        # Record experimental data in a dict
        d_log = {}
        d_log["nb_classes"] = nb_classes
        d_log["num_frames"] = num_frames
        d_log["batch_size"] = batch_size
        d_log["n_batch_per_epoch"] = n_batch_per_epoch
        d_log["nb_epoch"] = nb_epoch
        d_log["epoch_size"] = epoch_size
        d_log["prob"] = prob
        d_log["optimizer"] = opt.get_config()
        d_log["augmentator_config"] = DataAug.get_config()
        d_log["train_loss"] = list_train_loss
        d_log["test_loss"] = list_test_loss

        from keras.utils.visualize_util import plot
        png_file = os.path.join(exp_dir, 'archi.png')
        plot(model, to_file=png_file, show_shapes=True)

        json_file = os.path.join(exp_dir, 'experiment_log.json')
        general_utils.save_exp_log(json_file, d_log)

    except KeyboardInterrupt:
        shutil.rmtree(exp_dir)

if __name__ == '__main__':

    pass
