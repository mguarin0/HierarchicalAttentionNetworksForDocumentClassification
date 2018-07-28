"""
@author: Michael Guarino
"""

import tensorflow as tf
import os
import csv
import re
import itertools
import more_itertools
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils import prjPaths

class IMDB:

  def __init__(self, action):
    """
    desc: this class is used to process the imdb dataset
    args:
        action: specify whether to create or fetch the data using the IMDB class
    """
    self.paths = prjPaths()
    self.ROOT_DATA_DIR = self.paths.ROOT_DATA_DIR
    self.DATASET = "imdb"
    
    self.CSVFILENAME = os.path.join(self.ROOT_DATA_DIR, self.DATASET, "{}.csv".format(self.DATASET))
    assert(action in ["create", "fetch"]), "invalid action"

    if action == "create":

      # if creating new csv remove old if one exists
      if os.path.exists(self.CSVFILENAME):
        print("removing existing csv file from {}".format(self.CSVFILENAME))
        os.remove(self.CSVFILENAME)

      # directory structure
      train_dir = os.path.join(self.ROOT_DATA_DIR, self.DATASET, "aclImdb", "train")
      test_dir = os.path.join(self.ROOT_DATA_DIR, self.DATASET, "aclImdb", "test")

      trainPos_dir = os.path.join(train_dir, "pos")
      trainNeg_dir = os.path.join(train_dir, "neg")

      testPos_dir = os.path.join(test_dir, "pos")
      testNeg_dir = os.path.join(test_dir, "neg")

      self.data = {"trainPos": self._getDirContents(trainPos_dir),
                   "trainNeg": self._getDirContents(trainNeg_dir),
                   "testPos": self._getDirContents(testPos_dir),
                   "testNeg": self._getDirContents(testNeg_dir)}
  # end

  def _getDirContents(self, path):
    """
    desc: get all filenames in a specified directory
    args:
      path: path of directory to get contents of 
    returns:
      dirFiles: list of filenames in a directory
    """
    dirFiles = os.listdir(path)
    dirFiles = [os.path.join(path, file) for file in dirFiles]
    return dirFiles
  # end

  def _getID_label(self, file, binary):
    """
    desc: get label for a specific filename
    args:
      file: current file being operated on 
      binary: specify if data should be recoded as binary or kept in original form for imdb dataset
    returns:
      list of unique identifier of file, label, and if it is test or training data
    """
    splitFile = file.split("/")
    testOtrain = splitFile[-3]
    filename = os.path.splitext(splitFile[-1])[0]
    id, label = filename.split("_")
    if binary:
      if int(label) < 5:
        label = 0
      else:
        label = 1

    return [id, label, testOtrain]
  # end

  def _loadTxtFiles(self, dirFiles, binary):
    """
    desc: load and format all imdb dataset
    args:
      dirFiles: current file being operated on
      binary: specify if data should be recoded as binary or kept in original form for imdb dataset
    returns:
      list of dictionaries containing all information about imdb dataset
    """
    TxtContents = list()
    for file in tqdm(dirFiles, desc="process all files in a directory"):
      try:
        with open(file, encoding="utf8") as txtFile:
          content = txtFile.read()
          id, label, testOtrain = self._getID_label(file, binary=binary)
          TxtContents.append({"id": id,
                              "content": content,
                              "label": label,
                              "testOtrain": testOtrain})
      except:
        print("this file threw and error and is being omited: {}".format(file))
        continue
    return TxtContents
  # end

  def _writeTxtFiles(self, TxtContents):
    """
    desc: write imdb content and meta data to csv 
    args:
      TxtContents: list of dictionaries containing all information about imdb dataset 
    """

    with open(self.CSVFILENAME, "a") as csvFile:
      fieldNames = ["id", "content", "label", "testOtrain"]
      writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
      writer.writeheader()

      for seq in TxtContents:
        try:
          writer.writerow({"id": seq["id"],
                           "content": seq["content"].encode("ascii", "ignore").decode("ascii"),
                           "label": seq["label"],
                           "testOtrain": seq["testOtrain"]})
        except:
          print("this sequence threw an exception: {}".format(seq["id"]))
          continue
  # end

  def createManager(self, binary):
    """
    desc: This function is called by create_csv.py script. 
          It manages the loading, formatting, and creation of a csv from the imdb directory structure.
    args:
      binary: specify if data should be recoded as binary or kept in original form for imdb dataset
    """

    for key in self.data.keys():
      self.data[key] = self._loadTxtFiles(self.data[key], binary)
      self._writeTxtFiles(self.data[key])
  # end

  def _clean_str(self, string):
    """
    desc: This function cleans a string
          adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    args:
      string: the string to be cleaned
    returns:
      a cleaned string
    """

    string = BeautifulSoup(string, "lxml").text
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split(" ")
  # end

  def _oneHot(self, ys):
    """
    desc: one hot encodes labels in dataset
    args: 
      ys: dataset labels
    returns:
      list of one hot encoded training, testing, and lookup labels 
    """

    y_train, y_test = ys
    y_train = list(map(int, y_train)) # confirm all type int
    y_test = list(map(int, y_test)) # confirm all type int
    lookuplabels = {v: k for k, v in enumerate(sorted(list(set(y_train + y_test))))}
    recoded_y_train = [lookuplabels[i] for i in y_train]
    recoded_y_test = [lookuplabels[i] for i in y_test]
    labels_y_train = tf.constant(recoded_y_train)
    labels_y_test = tf.constant(recoded_y_test)
    max_label = tf.reduce_max(labels_y_train + labels_y_test)
    labels_y_train_OHE = tf.one_hot(labels_y_train, max_label+1)
    labels_y_test_OHE = tf.one_hot(labels_y_test, max_label+1)

    with tf.Session() as sess:
      # Initialize all variables
      sess.run(tf.global_variables_initializer())
      #l = sess.run(labels)
      y_train_ohe = sess.run(labels_y_train_OHE)
      y_test_ohe = sess.run(labels_y_test_OHE)
      sess.close()
    return [y_train_ohe, y_test_ohe, lookuplabels]
  # end

  def _index(self, xs):
    """
    desc: apply index to text data and persist unique vocabulary in dataset to pickle file
    args:
      xs: text data 
    returns:
      list of test, train data after it was indexed, the lookup table for the vocabulary,
      and any persisted variables that may be needed
    """
    def _apply_index(txt_data):
      indexed = [[[unqVoc_LookUp[char] for char in seq] for seq in doc] for doc in txt_data]
      return indexed
    # end

    x_train, x_test = xs

    # create look up table for all unique vocab in test and train datasets
    unqVoc = set(list(more_itertools.collapse(x_train[:] + x_test[:])))
    unqVoc_LookUp = {k: v+1 for v, k in enumerate(unqVoc)}
    vocab_size = len(list(unqVoc_LookUp))

    x_train = _apply_index(txt_data=x_train)
    x_test = _apply_index(txt_data=x_test)

    # determine max sequence lengths
    max_seq_len = max([len(seq) for seq in itertools.chain.from_iterable(x_train + x_test)]) # max length of sequence across all documents
    max_sent_len = max([len(sent) for sent in (x_train + x_test)]) # max length of sentence across all documents
   
    persisted_vars = {"max_seq_len":max_seq_len,
                      "max_sent_len":max_sent_len,
                      "vocab_size":vocab_size}

    return [x_train, x_test, unqVoc_LookUp, persisted_vars]
  # end

  def partitionManager(self, dataset):
    """
    desc: apply index to text data, one hot encode labels, and persist unique vocabulary in dataset to pickle file
    args: 
      dataset: dataset to be processed
    returns:
      return list of indexed training, training data along with one hot encoded labels
    """
    assert(self.DATASET==dataset), "this function works on {} and is not meant to process {} dataset".format(self.DATASET, dataset)

    # load csv file
    df = pd.read_csv(self.CSVFILENAME)

    # partition data
    train = df.loc[df["testOtrain"] == "train"]
    test = df.loc[df["testOtrain"] == "test"]

    # create 3D list for han model and clean strings
    create3DList = lambda df: [[self._clean_str(seq) for seq in "|||".join(re.split("[.?!]", docs)).split("|||")]
                                                                for docs in df["content"].values]
    x_train = create3DList(df=train)
    x_test = create3DList(df=test)

    # index and persist unq vocab in pickle file
    x_train, x_test, unqVoc_LookUp, persisted_vars  = self._index(xs=[x_train[:], x_test[:]])

    y_train = train["label"].tolist()
    y_test = test["label"].tolist()

    #OHE classes
    y_train_ohe, y_test_ohe, lookuplabels = self._oneHot(ys=[y_train, y_test])

    # update persisted vars
    persisted_vars["lookuplabels"] = lookuplabels
    persisted_vars["num_classes"] = len(lookuplabels.keys())

    # save lookup table and variables that need to be persisted
    if not os.path.exists(os.path.join(self.paths.LIB_DIR, self.DATASET)):
      os.mkdir(os.path.join(self.paths.LIB_DIR, self.DATASET))
    pickle._dump(unqVoc_LookUp, open(os.path.join(self.paths.LIB_DIR, self.DATASET, "unqVoc_Lookup.p"), "wb"))
    pickle._dump(persisted_vars, open(os.path.join(self.paths.LIB_DIR, self.DATASET, "persisted_vars.p"), "wb"))

    return[x_train, y_train_ohe, x_test, y_test_ohe]
  # end

  def get_data(self, type_):
    """
    desc: load and return dataset from binary files
    args:
      type_: type of dataset (train, val, test)
    returns:
      loaded dataset
    """

    assert(type_ in ["train", "val", "test"])

    print("loading {} dataset...".format(type_))

    x = np.load(os.path.join(self.paths.ROOT_DATA_DIR, self.DATASET, "{}_x.npy".format(type_)))
    y = np.load(os.path.join(self.paths.ROOT_DATA_DIR, self.DATASET, "{}_y.npy".format(type_)))
    docsize = np.load(os.path.join(self.paths.ROOT_DATA_DIR, self.DATASET, "{}_docsize.npy".format(type_)))
    sent_size = np.load(os.path.join(self.paths.ROOT_DATA_DIR, self.DATASET, "{}_sent_size.npy".format(type_)))
    return [x, y, docsize, sent_size]
  # end

  def get_batch_iter(self, data, batch_size, num_epochs, shuffle=True):
    """
    desc: batch dataset generator
    args:
      data: dataset to batch as list
      batch_size: the batch size used
      num_epochs: number of training epochs
      shuffle: shuffle dataset
    returns:
    adapted from Denny Britz https://github.com/dennybritz/cnn-text-classification-tf.git
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
      # Shuffle the data at each epoch
      if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        next_batch = data[shuffle_indices]
      else:
        next_batch = data
      for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        #yield next_batch[start_index:end_index]
        yield epoch, next_batch[start_index:end_index]
  # end

  def hanformater(self, inputs):
    """
    desc: format data specific for hierarchical attention networks
    args:
      inputs: data
    returns:
      dataset with corresponding dimensions for document and sentence level
    """

    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))

    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD

    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(tqdm(inputs, desc="formating data for hierarchical attention networks")):
      for j, sentence in enumerate(document):
        sentence_sizes[i, j] = sentence_sizes_[i][j]
        for k, word in enumerate(sentence):
          b[i, j, k] = word
    return b, document_sizes, sentence_sizes
  # end
# end
