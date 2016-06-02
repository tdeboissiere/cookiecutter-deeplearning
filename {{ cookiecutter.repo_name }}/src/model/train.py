import sys
import os
import models
from dotenv import load_dotenv, find_dotenv
sys.path.append("../utils")
import general_utils


def train_VGG(**kwargs):

    # Set default params
    d_params = {"nb_classes": 2,
                "num_frames": 5,
                "batch_size": 32,
                "n_batch_per_epoch": 100,
                "nb_epoch": 5,
                "prob": 0.5,
                "do_train": True,
                "do_plot": False,
                "data_file": None
                }

    # Update keyword if specified
    for key in kwargs.keys():
        try:
            d_params[key] = kwargs[key]
        except KeyError:
            print "Wrong kwarg"
            print "Valid are: %s" % " ".join(d_params.keys())
            sys.exit()

    if d_params["do_train"]:
        model = models.VGG(d_params["nb_classes"])
        # Launch training
        models.train(model, **d_params)

    do_plot_result = False
    if do_plot_result:
        general_utils.plot_results()


if __name__ == "__main__":

    # Set default params
    d_params = {"eod_type":"full_frame",
                "goal": "binclf_sleep",
                "nb_classes": 2,
                "batch_size": 32,
                "n_batch_per_epoch": 1000,
                "nb_epoch": 20,
                "prob": 0.6,
                "do_train": True,
                "do_plot": False
                }

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Load env variables
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))

    #############################################
    # Train VGG
    #############################################
    data_file = ""
    data_file = os.path.join(data_dir, data_file)
    d_VGG = d_params.copy()
    d_VGG["data_file"] = data_file
    train_VGG(**d_VGG)
