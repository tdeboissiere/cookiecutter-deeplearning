import os
import sys
import h5py
import cPickle as pickle

from kerasdeconv.KerasDeconv import DeconvNet
from kerasdeconv.utils import get_deconv_images
from kerasdeconv.utils import plot_max_activation
from keras.models import model_from_json

from dotenv import load_dotenv, find_dotenv

sys.path.append("../utils")
import general_utils


def load_model(model_archi_file, weights_file):
    """

    args: model_path (str) dir where weights and archi are stored

    returns model (Keras model)
    """

    # model reconstruction from JSON:
    print "Load archi"
    model = model_from_json(open(model_archi_file).read())
    print "Load weights"
    model.load_weights(weights_file)
    print "Compile"
    model.compile(optimizer="adagrad", loss="categorical_crossentropy")

    return model


if __name__ == "__main__":

    ######################
    # Misc
    ######################

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())
    # Load env variables
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))
    fig_dir = os.path.expanduser(os.environ.get("FIG_DIR"))
    fig_dir = os.path.join(fig_dir, "SFCNN_deconv")
    general_utils.create_dir(fig_dir)

    #############################################
    # Train VGG
    #############################################

    # Get data, model, weights paths
    data_file = ""
    data_path = os.path.join(data_dir, data_file)
    exp_path = ""
    weights_file = os.path.join(exp_path, "")
    model_archi_file = os.path.join(exp_path, "SFCNN_archi.json")

    # Load data
    with h5py.File(data_path, "r") as hf:
        X_test = hf["train_data"][:, :, :, :]

    ###############################################
    # Get max activation for a slection of feat maps
    ###############################################
    get_max_act = True
    if get_max_act:
        model = load_model(model_archi_file, weights_file)
        Dec = DeconvNet(model)
        d_act_path = os.path.join(fig_dir, "dict_top9_mean_act.pickle")
        d_act = {"convolution2d_10": {}
                 }
        for feat_map in range(64):
            print "%s/%s" % (feat_map + 1, 64)
            d_act["convolution2d_10"][feat_map] = Dec.find_top9_mean_act(
                X_test, "convolution2d_10", feat_map, batch_size=64)
        with open(d_act_path, "w") as f:
            pickle.dump(d_act, f)

    ###############################################
    # Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    deconv_img = True
    if deconv_img:
        d_act_path = os.path.join(fig_dir, "dict_top9_mean_act.pickle")
        d_deconv_path = os.path.join(fig_dir, "dict_top9_deconv")
        if not model:
            model = load_model(model_archi_file, weights_file)
        if not Dec:
            Dec = DeconvNet(model)
        get_deconv_images(d_act_path, d_deconv_path, X_test, Dec)

    ###############################################
    # Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    plot_deconv_img = True
    if plot_deconv_img:
        d_act_path = os.path.join(fig_dir, "dict_top9_mean_act.pickle")
        d_deconv_path = os.path.join(fig_dir, "dict_top9_deconv.npz")
        target_layer = "convolution2d_10"
        plot_max_activation(d_act_path, d_deconv_path, X_test,
                            target_layer, save=True, save_path=fig_dir)
