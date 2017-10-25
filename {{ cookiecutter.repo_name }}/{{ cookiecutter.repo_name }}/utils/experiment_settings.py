import os
from . import logging_utils as lu


class ExperimentSettings(object):

    def __init__(self, cli_args):

        # Transfer attributes
        self.__dict__.update(cli_args.__dict__)

    def setup_dir(self):

        for path in ["figures", "data/processed", "models"]:
            if not os.path.exists(path):
                os.makedirs(path)

        self.fig_dir = "figures"
        self.data_dir = "data/processed"
        self.model_dir = "models"
