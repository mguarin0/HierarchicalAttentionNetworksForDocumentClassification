"""
@author: Michael Guarino
"""

import os
import shutil
import platform
import urllib.request
import tarfile
import traceback
import argparse

from utils import prjPaths

def get_args():
  """
  desc: get cli arguments
  returns:
    args: dictionary of cli arguments
  """

  parser = argparse.ArgumentParser(description="this script is used for downloading datasets for training this implementation of the Hierarchical Attention Networks")
  parser.add_argument("dataset", choices=["imdb"], default="imdb", help="dataset to use", type=str)
  args = parser.parse_args()
  return args
# end

def download(paths, args):
  """
  desc: download a dataset from url
  args:
    args: dictionary of cli arguments
    paths: project paths 
  """

  if args.dataset == "imdb":
    resource_loc = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    osType = platform.system()
    if osType == "Windows":
      print("manually download data set from {}"\
            " and set getDataset=False when prjPaths is called in *_master.py script".format(resource_loc))
      exit(0)
    elif osType is not "Linux":
      osType = "OSX"

    filename=os.path.join(paths.ROOT_DATA_DIR, args.dataset, "aclImdb_v1.tar.gz")
    ACLIMDB_DIR = os.path.join(paths.ROOT_DATA_DIR, args.dataset)

    # if tar file already exists remove it
    if os.path.exists(filename):
      os.remove(filename)
    # if fclImdb dir already exists remove it
    if os.path.exists(os.path.join(ACLIMDB_DIR, "aclImdb")):
      shutil.rmtree(os.path.join(ACLIMDB_DIR, "aclImdb"))
    else:
      os.mkdir(ACLIMDB_DIR)

    print("downloading: {}".format(args.dataset))
    try:
      urllib.request.urlretrieve(resource_loc, filename)
    except Exception as e:
      print("something went wrong downloading: {} at {}".format(args.dataset, resource_loc))
      traceback.print_exc()

    print("unpacking: {}".format(args.dataset))
    if (filename.endswith("tar.gz")):
      tar = tarfile.open(filename, "r:gz")
      tar.extractall(ACLIMDB_DIR)
      tar.close()
    elif (filename.endswith("tar")):
      tar = tarfile.open(filename, "r:")
      tar.extractall(ACLIMDB_DIR)
      tar.close()
# end

if __name__ == "__main__":
  paths = prjPaths()
  args = get_args()
  download(paths=paths, args=args)
  print("download complete!")
