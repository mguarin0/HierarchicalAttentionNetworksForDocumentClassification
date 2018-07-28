"""
@author: Michael Guarino
"""

import os
import datetime
import logging

class prjPaths:
  def __init__(self):
    """
    desc: create object containing project paths
    """

    self.SRC_DIR = os.path.abspath(os.path.curdir)
    self.ROOT_MOD_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
    self.ROOT_DATA_DIR = os.path.join(self.ROOT_MOD_DIR, "data")
    self.LIB_DIR = os.path.join(self.ROOT_MOD_DIR, "lib")
    self.CHECKPOINT_DIR = os.path.join(self.LIB_DIR, "chkpts")
    self.SUMMARY_DIR = os.path.join(self.LIB_DIR, "summaries")
    self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")

    pth_exists_else_mk = lambda path: os.mkdir(path) if not os.path.exists(path) else None
        
    pth_exists_else_mk(self.ROOT_DATA_DIR)
    pth_exists_else_mk(self.LIB_DIR)
    pth_exists_else_mk(self.CHECKPOINT_DIR)
    pth_exists_else_mk(self.SUMMARY_DIR)
    pth_exists_else_mk(self.LOGS_DIR)
  # end
# end

def get_logger(paths):
  # TODO logger not logging to file
  currentTime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
  logFileName = os.path.join(paths.LOGS_DIR, "HAN_TxtClassification_{}.log".format(currentTime))

  logger = logging.getLogger(__name__)
  formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

  fileHandler = logging.FileHandler(logFileName)
  fileHandler.setLevel(logging.INFO)
  fileHandler.setFormatter(formatter)

  logger.addHandler(fileHandler)

  return logger
# end

