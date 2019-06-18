#Usage:
#python distribute_data.py -i /home/ray/tensorflow/workspace/kayaker_ssd/images/ -p 0.2

import os
import glob
import argparse
import re
from shutil import copyfile,move
import xml.etree.ElementTree as ET
import panda as pd


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="distribute training and testing data")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .xml files are stored",
                        type=str)
    parser.add_argument("-p",
                        "--percentage",
                        help="testing/training",
                        type=str)
    args = parser.parse_args()

    if(args.inputDir is None):
        return

    if(args.percentage is None):
        args.percentage = 0.2
    
    assert(os.path.isdir(args.inputDir))
    args.inputDir += '/'
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    files = glob.glob(args.inputDir + 'annotation/*.xml')
    dataset = []
    dataset.append([])
    dataset.append([])
    dataset.append([])
    dataset.append([])
    dataset.append([])

    for i,xml_file in enumerate(files):
        dataset[i%5].append(xml_file)
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    
    try:
        os.mkdir(args.inputDir+'set0')
        os.mkdir(args.inputDir+'set1')
        os.mkdir(args.inputDir+'set2')
        os.mkdir(args.inputDir+'set3')
        os.mkdir(args.inputDir+'set4')
    except:
        print 'test folder already esxist'
        
    for set in dataset:
        for i,xml_file in enumerate(set):
            name = os.path.splitext(os.path.basename(xml_file))
            jpg_file = args.inputDir + 'image/' +name[0]+'.jpg'
            #print jpg_file
            move(xml_file,args.inputDir + 'set%d/'%(i%5)+name[0]+name[1])
            move(jpg_file,args.inputDir + 'set%d/'%(i%5)+name[0]+'.jpg')
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    
if __name__ == '__main__':
    main()