"""
@author: Michael Guarino
"""

import tensorflow as tf
import numpy as np
import argparse
import time
import os
import sys
import pickle
from tqdm import tqdm
from dataProcessing import IMDB
from utils import prjPaths

def get_args():
  """
  desc: get cli arguments
  returns:
    args: dictionary of cli arguments
  """

  parser = argparse.ArgumentParser(description="this script creates tf record files",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("dataset", choices=["imdb"], default="imdb", help="dataset to use", type=str)
  parser.add_argument("--train_data_percentage", default=0.70, help="percent of dataset to use for training", type=float)
  parser.add_argument("--validation_data_percentage", default=0.20,  help="percent of dataset to use for validation", type=float)
  parser.add_argument("--test_data_percentage", default=0.10,  help="percent of dataset to use for testing", type=float)
  args = parser.parse_args()
  return args
# end

def _write_binaryfile(nparray, filename):
  """
  desc: write dataset partition to binary file
  args:
    nparray: dataset partition as numpy array to write to binary file 
    filename: name of file to write dataset partition to
  """

  np.save(filename, nparray)
# end

def serialize_data(paths, args):
  """
  desc: write dataset partition to binary file
  args:
    nparray: dataset partition as numpy array to write to binary file 
    filename: name of file to write dataset partition to
  """

  if args.dataset == "imdb":

    # fetch imdb dataset
    imdb = IMDB(action="fetch")
    tic = time.time() # start time of data fetch
    x_train, y_train, x_test, y_test = imdb.partitionManager(args.dataset)

    toc = time.time() # end time of data fetch
    print("time taken to fetch {} dataset: {}(sec)".format(args.dataset, toc - tic))

    # kill if shapes don't make sense
    assert(len(x_train) == len(y_train)), "x_train length does not match y_train length"
    assert(len(x_test) == len(y_test)), "x_test length does not match y_test length"

    # combine datasets 
    x_all = x_train + x_test
    y_all = np.concatenate((y_train, y_test), axis=0)

    # create slices
    train_slice_lim = int(round(len(x_all)*args.train_data_percentage))
    validation_slice_lim = int(round((train_slice_lim) + len(x_all)*args.validation_data_percentage))

    # partition dataset into train, validation, and test sets
    x_all, docsize, sent_size = imdb.hanformater(inputs=x_all)

    x_train = x_all[:train_slice_lim]
    y_train = y_all[:train_slice_lim]
    docsize_train = docsize[:train_slice_lim]
    sent_size_train = sent_size[:train_slice_lim]

    x_val = x_all[train_slice_lim+1:validation_slice_lim]
    y_val = y_all[train_slice_lim+1:validation_slice_lim]
    docsize_val = docsize[train_slice_lim+1:validation_slice_lim]
    sent_size_val = sent_size[train_slice_lim+1:validation_slice_lim]


    x_test = x_all[validation_slice_lim+1:]
    y_test = y_all[validation_slice_lim+1:]
    docsize_test = docsize[validation_slice_lim+1:]
    sent_size_test = sent_size[validation_slice_lim+1:]

    train_bin_filename_x = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "train_x.npy")
    train_bin_filename_y = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "train_y.npy")
    train_bin_filename_docsize = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "train_docsize.npy")
    train_bin_filename_sent_size = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "train_sent_size.npy")

    val_bin_filename_x = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "val_x.npy")
    val_bin_filename_y = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "val_y.npy")
    val_bin_filename_docsize = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "val_docsize.npy")
    val_bin_filename_sent_size = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "val_sent_size.npy")

    test_bin_filename_x = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "test_x.npy")
    test_bin_filename_y = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "test_y.npy")
    test_bin_filename_docsize = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "test_docsize.npy")
    test_bin_filename_sent_size = os.path.join(paths.ROOT_DATA_DIR, args.dataset, "test_sent_size.npy")

    _write_binaryfile(nparray=x_train, filename=train_bin_filename_x)
    _write_binaryfile(nparray=y_train, filename=train_bin_filename_y)
    _write_binaryfile(nparray=docsize_train, filename=train_bin_filename_docsize)
    _write_binaryfile(nparray=sent_size_train, filename=train_bin_filename_sent_size)

    _write_binaryfile(nparray=x_val, filename=val_bin_filename_x)
    _write_binaryfile(nparray=y_val, filename=val_bin_filename_y)
    _write_binaryfile(nparray=docsize_val, filename=val_bin_filename_docsize)
    _write_binaryfile(nparray=sent_size_val, filename=val_bin_filename_sent_size)

    _write_binaryfile(nparray=x_test, filename=test_bin_filename_x)
    _write_binaryfile(nparray=y_test, filename=test_bin_filename_y)
    _write_binaryfile(nparray=docsize_test, filename=test_bin_filename_docsize)
    _write_binaryfile(nparray=sent_size_test, filename=test_bin_filename_sent_size)
# end

if __name__ == "__main__":
  paths = prjPaths()
  args = get_args()
  serialize_data(paths, args=args)
