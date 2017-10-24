import sys
import os
import pandas as pd
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix

from dotenv import load_dotenv, find_dotenv
try:
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import cm
except:
    pass
import json
import glob



def pretty_print(string):
    """
    Simple utility to highlight printing

    args: string (str) string to print
    """

    print("")
    print(string)
    print("")


def remove_files(files):
    """
    Remove files from disk

    args: files (str or list) remove all files in 'files'
    """

    if isinstance(files, (list, tuple)):
        for f in files:
            if os.path.isfile(os.path.expanduser(f)):
                os.remove(f)
    elif isinstance(files, str):
        if os.path.isfile(os.path.expanduser(files)):
            os.remove(files)


def create_dir(dirs):
    """
    Create directory

    args: dirs (str or list) create all dirs in 'dirs'
    """

    if isinstance(dirs, (list, tuple)):
        for d in dirs:
            if not os.path.exists(os.path.expanduser(d)):
                os.makedirs(d)
    elif isinstance(dirs, str):
        if not os.path.exists(os.path.expanduser(dirs)):
            os.makedirs(dirs)


def plot_batch_adversarial(X, y, batch_size):
    """
    Plot the images in X, add a label (in y)

    details: build a gridspec of the size of the batch
             (valid batch_sizes are multiple of 2 from 8 to 64)
    """

    d_class = {0: "Training set",
               1: "Test set"}

    assert X.shape[0] >= batch_size, "batch size greater than X.shape[0]"

    if batch_size == 8:
        gs = gridspec.GridSpec(2, 4)
    elif batch_size == 16:
        gs = gridspec.GridSpec(4, 4)
    elif batch_size == 32:
        gs = gridspec.GridSpec(4, 8)
    elif batch_size == 64:
        gs = gridspec.GridSpec(8, 8)
    else:
        print("Batch too big")
        return
    fig = plt.figure(figsize=(15, 15))
    for i in range(batch_size):
        ax = plt.subplot(gs[i])
        img = X[i, :, :, :]
        img_shape = img.shape
        min_s = min(img_shape)
        if img_shape.index(min_s) == 0:
            img = img.transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_xlabel(d_class[int(y[i])], fontsize=8)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    gs.tight_layout(fig)
    plt.show()
    raw_input()
    plt.clf()
    plt.close()


def plot_batch(X, y, batch_size):
    """
    Plot the images in X, add a label (in y)

    details: build a gridspec of the size of the batch
             (valid batch_sizes are multiple of 2 from 8 to 64)
    """

    d_class = {0: "safe driving",
               1: "texting - right",
               2: "talking on the phone - right",
               3: "texting - left",
               4: "talking on the phone - left",
               5: "operating the radio",
               6: "drinking",
               7: "reaching behind",
               8: "hair and makeup",
               9: "talking to passenger"}

    assert X.shape[0] >= batch_size, "batch size greater than X.shape[0]"

    if batch_size == 8:
        gs = gridspec.GridSpec(2, 4)
    elif batch_size == 16:
        gs = gridspec.GridSpec(4, 4)
    elif batch_size == 32:
        gs = gridspec.GridSpec(4, 8)
    elif batch_size == 64:
        gs = gridspec.GridSpec(8, 8)
    else:
        print("Batch too big")
        return
    fig = plt.figure(figsize=(15, 15))
    for i in range(batch_size):
        ax = plt.subplot(gs[i])
        img = X[i, :, :, :]
        img_shape = img.shape
        min_s = min(img_shape)
        if img_shape.index(min_s) == 0:
            img = img.transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_xlabel(d_class[int(y[i])], fontsize=8)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    gs.tight_layout(fig)
    plt.show()
    raw_input()
    plt.clf()
    plt.close()


def plot_batch_3D(X, y, batch_size):
    """
    Plot a few of the images in X, add a label (in y)
    X should be a video (n_samples, n_channel, n_frames, h, w)

    details: build a gridspec where all the frames of a single video
             are shown.
             Plot only a few videos (currently two, else the plot gets saturated)
    """
    d_class = {0: "False P", 1: "True Sleep"}

    gs = gridspec.GridSpec(4, 8)
    fig = plt.figure(figsize=(15, 15))
    for i in range(8):
        ax = plt.subplot(gs[0, i])
        ax.imshow(X[0, 0, i, :, :], cmap="Greys_r")
        ax.set_xlabel("%s" % (d_class[int(y[0])]), fontsize=15)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    for i in range(8):
        ax = plt.subplot(gs[1, i])
        ax.imshow(X[0, 0, i + 7, :, :], cmap="Greys_r")
        ax.set_xlabel("%s" % (d_class[int(y[0])]), fontsize=15)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    for i in range(8):
        ax = plt.subplot(gs[2, i])
        ax.imshow(X[1, 0, i, :, :], cmap="Greys_r")
        ax.set_xlabel("%s" % (d_class[int(y[1])]), fontsize=15)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    for i in range(8):
        ax = plt.subplot(gs[3, i])
        ax.imshow(X[1, 0, i + 7, :, :], cmap="Greys_r")
        ax.set_xlabel("%s" % (d_class[int(y[1])]), fontsize=15)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    gs.tight_layout(fig)
    plt.show()
    raw_input()


def plot_confusion_matrix(y_true, y_pred, list_class, cmap='Blues'):
    """
    Plot the confusions matrix

    args:  y_true (np array) the true class labels
           y_pred (np array) the predicted class labels
           list_class (list) the list of class labels
           cmap (str) matplotlib color map
    """

    # Compute confusion matrix
    confM = confusion_matrix(y_true, y_pred)
    # Normalise it
    confM = confM.astype('float') / confM.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)

    # Map the list of classes to strings
    list_class = map(str, list_class)

    plt.imshow(confM, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(list_class))
    plt.xticks(tick_marks, list_class, rotation=45)
    plt.yticks(tick_marks, list_class)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    raw_input()


def plot_clf_distribution(y_true, y_pred_proba):
    """
    Plot the histogram of the predictions (with color code) to identify classes

    args:  y_true (np array) the true class labels
           y_pred_proba (np array) the predicted class probabilities
    """

    y_pred_0 = y_pred_proba[y_true == 0, 1]
    y_pred_1 = y_pred_proba[y_true == 1, 1]

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2, 1)
    # Train
    ax_0 = plt.subplot(gs[0])
    _, bins0, _ = ax_0.hist(y_pred_0, bins=100, histtype="stepfilled", color="steelblue")
    ax_0.set_xlabel("Class 0", fontsize=22)
    # Test
    ax_1 = plt.subplot(gs[1])
    _, bins1,_ = ax_1.hist(y_pred_1, bins=100, histtype="stepfilled", color="orangered")
    ax_1.set_xlabel("Class 1", fontsize=22)
    # Set axis range
    xmin = np.min(bins0.tolist() + bins1.tolist())
    xmax = np.max(bins0.tolist() + bins1.tolist())
    ax_0.set_xlim([xmin, xmax])
    ax_1.set_xlim([xmin, xmax])
    ax_0.set_yscale("log")
    ax_1.set_yscale("log")
    gs.tight_layout(fig)
    plt.show()
    raw_input()


def plot_model_clf_dist(data_path, model_archi_file, weights_file):
    """
    Plot the histogram of class predictions

    details: Load test data in data_path, use model specified by model_archi_file
             and load trained weights specified in weights_file

    args: data_path (str) the path to the HDF5 file holding the data
          model_archi_file (str) the path to the json file holding the model archi
          weights_file (str) the path to the HDF5 file holding the weights
    """

    with h5py.File(data_path, "r") as hf:
        if "C3D" in model_archi_file:
            X_test, y_test = hf["train_data"][:1000,:, :, :, :], hf["train_label"][:1000]
            X_test = X_test.transpose(0, 2, 1, 3, 4)
            X_test = X_test / 255. - 0.5
        else:
            X_test, y_test = hf["test_data"][:, 0, :, :, :], hf["test_label"][:]
            X_test = X_test / 255. - 0.5

        # model reconstruction from JSON:
        from keras.models import model_from_json
        print("Load archi")
        model = model_from_json(open(model_archi_file).read())
        print("Load weights")
        model.load_weights(weights_file)
        print("Compile")
        model.compile(optimizer="adagrad", loss="categorical_crossentropy")
        print("Predict")
        y_pred_proba = model.predict(X_test, verbose=0)
        plot_clf_distribution(y_test, y_pred_proba)


def save_exp_log(file_name, d_log):
    """
    Utility to save the experiment log as json file

    details: Save the experiment log (a dict d_log) in file_name

    args: file_name (str) the file in which to save the log
          d_log (dict) python dict holding experiment log
    """

    with open(file_name, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)


def plot_results():
    """
    Utility to compare the results of several experiments (in terms) of loss

    (WIP)
    """
    list_exp = glob.glob("./Log/*")
    list_d_log = []
    list_archi = []
    for exp_dir in list_exp:
        with open(exp_dir + "/experiment_log.json", "r") as f:
            d = json.load(f)
            list_d_log.append(d)
        with open(exp_dir + "/SFCNN_archi.json", "r") as f:
            d = json.load(f)
            list_layers = d["config"]
            list_conv = [l["config"]["name"] for l in list_layers
                         if l["class_name"] == "Convolution2D"]
            max_conv = max([int(name.split("_")[-1]) for name in list_conv])
            list_archi.append(max_conv)

    for i in range(len(list_exp)):
        d = list_d_log[i]
        max_conv = list_archi[i]
        d["max_conv"] = max_conv
        list_d_log[i] = d

    list_c = cm.Accent(np.linspace(0,1,10))

    list_by_batch = []
    list_by_aug = []
    list_by_epoch = []
    list_by_depth = []

    for d in list_d_log:
        if d["nb_epoch"] == 10 and \
                d["augmentator_config"]["transforms"] == {} and\
                d["max_conv"] == 4:
            list_by_batch.append(d)
        if d["nb_epoch"] > 10:
            list_by_epoch.append(d)
        if d["nb_epoch"] == 10 and \
                d["augmentator_config"]["transforms"] != {}and\
                d["max_conv"] == 4:
            list_by_aug.append(d)
        if d["nb_epoch"] == 10 and \
                d["augmentator_config"]["transforms"] == {} and \
                d["batch_size"] == 32:
            list_by_depth.append(d)

    # plt.figure(figsize=(12, 9))
    # c_counter = 0
    # for d in list_by_depth:

    #         # Legend
    #     label = "Batch size: %s" % d["batch_size"]
    #     label = "Data augmentation prob: %s" % d["prob"]
    #     label = "CNN depth: %s" % d["max_conv"]

    #     plt.plot(d["train_loss"], "--",
    #              label=label,
    #              color=list_c[c_counter],
    #              linewidth=3)
    #     plt.plot(d["test_loss"],
    #              label=label,
    #              color=list_c[c_counter],
    #              linewidth=3)
    #     c_counter += 1

    # plt.xlabel("Number of epochs", fontsize=18)
    # plt.ylabel("Logloss", fontsize=18)
    # plt.legend(loc="best")
    # plt.ylim([0.1, 0.8])
    # plt.tight_layout()
    # plt.show()
    # raw_input()

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(15, 15))
    list_labels = ["Batch size: ",
                   "Data augmentation prob: ",
                   "CNN depth: ",
                   "More epochs"
                   ]
    ll_d = [list_by_batch, list_by_aug, list_by_depth, list_by_epoch]
    for i in range(4):
        ax = plt.subplot(gs[i])
        c_counter = 0
        for d in ll_d[i]:
            if "Batch" in list_labels[i]:
                label = list_labels[i] + str(d["batch_size"])
            elif "augmentation" in list_labels[i]:
                label = list_labels[i] + str(d["prob"])
            elif "depth" in list_labels[i]:
                label = list_labels[i] + str(d["max_conv"])
            else:
                label = list_labels[i]
            ax.plot(d["train_loss"], "--",
                    label=label,
                    color=list_c[c_counter],
                    linewidth=3)
            ax.plot(d["test_loss"],
                    label=label,
                    color=list_c[c_counter],
                    linewidth=3)
            c_counter += 1
            ax.set_xlabel("Number of epochs", fontsize=18)
            ax.set_ylabel("Logloss", fontsize=18)
            ax.legend(loc="best")
            ax.set_ylim([0.1, 0.8])
            ax.text(0.05, 0.05, "Dashed: Training sample\nContinuous: Test sample",
                    transform=ax.transAxes,
                    fontsize=18,
                    bbox=dict(boxstyle='round', facecolor="white"))
    gs.tight_layout(fig)
    plt.savefig("./Figures/training_results.png")
    plt.show()
    raw_input()


def pickle3_to_pickle2():

    import pickle
    list_f = ["../../models/ResNet/Experiment_5/resnet_weights_fold%s.pickle" % i for i in range(8)]
    list_fpyth2 = ["../../models/ResNet/resnet_weights_fold%s_python2.pickle" % i for i in range(8)]
    for f, fout in zip(list_f, list_fpyth2):

        with open(f, "rb") as fp:
            d = pickle.load(fp)
            print(d.keys())
            print(d["values"][0][0][0])
            with open(fout, "wb") as fpyth2:
                pickle.dump(d, fpyth2, protocol=2)

def setup_logging(experiment):

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    model_dir = os.path.expanduser(os.environ.get("MODEL_DIR"))
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    # Output path where we store experiment log and weights
    model_dir = os.path.join(model_dir, "DCGAN")
    # Create if it does not exist
    create_dir(model_dir)
    # Automatically determine experiment name
    list_exp = glob.glob(model_dir + "/*")
    # Create the experiment dir and weights dir
    if experiment:
        exp_dir = os.path.join(model_dir, experiment)
    else:
        exp_dir = os.path.join(model_dir, "Experiment_%s" % len(list_exp))
    create_dir(exp_dir)

if __name__ == '__main__':

    pickle3_to_pickle2()
