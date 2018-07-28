"""
@author: Michael Guarino
"""

import os
import argparse

def get_args():
  """
  desc: get cli arguments
  returns:
    args: dictionary of cli arguments
  """

  parser = argparse.ArgumentParser(description="this script is used to download and process all data")
  parser.add_argument("dataset", choices=["imdb"], default="imdb", help="dataset to use", type=str)
  parser.add_argument("binary", default=True, help="coerce to binary classification", type=bool)
  args = parser.parse_args()
  return args
# end

if __name__ == "__main__":

  args = get_args()
  os.system("python3 download.py {}".format(args.dataset))
  os.system("python3 create_csv.py {} {}".format(args.dataset, args.binary))
  os.system("python3 serialize_data.py {}".format(args.dataset))
