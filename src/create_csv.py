"""
@author: Michael Guarino
"""

import os
import argparse
from dataProcessing import IMDB
from utils import prjPaths

def get_args():
  """
  desc: get cli arguments
  returns:
    args: dictionary of cli arguments
  """

  parser = argparse.ArgumentParser(description="this script is used for creating csv datasets for training this implementation of the Hierarchical Attention Networks")
  parser.add_argument("dataset", choices=["imdb"], default="imdb", help="dataset to use", type=str)
  parser.add_argument("binary", default=True, help="coerce to binary classification", type=bool)
  args = parser.parse_args()
  return args
# end

def create_csv(paths, args):
  """
  desc: This function creates a csv file from a downloaded dataset.
        Currently this process works on the imdb dataset but other datasets
        can be easily added.
  args:
    args: dictionary of cli arguments
    paths: project paths
  """

  if args.dataset == "imdb":
    print("creating {} csv".format(args.dataset))
    imdb = IMDB(action="create")
    imdb.createManager(args.binary)
    print("{} csv created".format(args.dataset))
# end

if __name__ == "__main__":
  paths = prjPaths()
  args = get_args()
  create_csv(paths=paths, args=args)
