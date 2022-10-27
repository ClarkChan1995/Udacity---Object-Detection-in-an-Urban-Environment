import argparse
import glob
import os
import random
import shutil
import numpy as np

from utils import get_module_logger


def split():
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    main_dic = "data/processed/"
    train_folder = "data/train/"
    test_folder = "data/test/"
    val_folder = "data/val/"

    # create folder for each file, TODO: Create it if not exists
    os.mkdir(train_folder)
    os.mkdir(test_folder)
    os.mkdir(val_folder)

    filenames = os.listdir(main_dic)
    
    random.shuffle(filenames)

    # split the file to 80% train, 10% test, 10% val
    train = int(0.8 * len(filenames))
    test = int(0.1 * len(filenames))
    val = int(0.1 * len(filenames))

    # move the files to each folder
    for i, filename in enumerate(filenames):
        if i < train:
            shutil.move(main_dic + filename, train_folder + filename)
        elif i >= train and i < train+test:
            shutil.move(main_dic + filename, test_folder + filename)
        else:
            shutil.move(main_dic + filename, val_folder + filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split()