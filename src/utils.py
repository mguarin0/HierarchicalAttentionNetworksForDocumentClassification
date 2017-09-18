"""
@author: Michael Guarino
"""

import os
import subprocess
import platform

class prjPaths:
    def __init__(self, getDataset=True):
        self.SRC_DIR = os.path.abspath(os.path.curdir)
        self.ROOT_MOD_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.ROOT_DATA_DIR = os.path.join(self.ROOT_MOD_DIR, "data")
        self.LIB_DIR = os.path.join(self.ROOT_MOD_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(self.LIB_DIR, "chkpts")
        self.CHECKPOINTS_HAN = os.path.join(self.CHECKPOINT_DIR, "han_chkpts/")
        self.CHECKPOINTS_HAN_GRU = os.path.join(self.CHECKPOINT_DIR, "han_gru_chkpts/")
        self.CHECKPOINTS_CNN = os.path.join(self.CHECKPOINT_DIR, "cnn_chkpts/")
        self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")

        if getDataset:
            osType = platform.system()
            if osType == "Windows":
                print("manually download data set from 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'"
                      " and set getDataset=False when prjPaths is called in *_master.py script")
                exit(0)
            elif osType is not "Linux":
                osType = "OSX"

            if not os.path.exists(self.ROOT_DATA_DIR):
                os.mkdir(path=self.ROOT_DATA_DIR)
            subprocess.Popen("sh getIMDB.sh {}".format(osType), shell=True, stdout=subprocess.PIPE).wait()
    # end
# end
