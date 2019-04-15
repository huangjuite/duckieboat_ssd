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
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    files = glob.glob(args.inputDir + 'annotation/*.xml')
    train_xmls = []
    test_xmls = []
    for i,xml_file in enumerate(files):
        if i<len(files)*args.percentage:
            test_xmls.append(xml_file)
        else:
            train_xmls.append(xml_file)
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    try:
        os.mkdir(args.inputDir+'test')
    except:
        print 'test folder already esxist'
    for xml_file in test_xmls:
        num = re.findall(r'\d+',xml_file)[-1]
        jpg_file = args.inputDir + 'image/frame'+num+'.jpg'

        move(xml_file,args.inputDir + 'test/frame'+num+'.xml')
        move(jpg_file,args.inputDir + 'test/frame'+num+'.jpg')
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    try:
        os.mkdir(args.inputDir+'train')
    except:
        print 'train folder already esxist'
    for xml_file in train_xmls:
        num = re.findall(r'\d+',xml_file)[-1]
        jpg_file = args.inputDir + 'image/frame'+num+'.jpg'

        move(xml_file,args.inputDir + 'train/frame'+num+'.xml')
        move(jpg_file,args.inputDir + 'train/frame'+num+'.jpg')

    print('Successfully transfer %d labeled sets'%len(files))
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    files = glob.glob(args.inputDir + 'image/*.jpg')
    print('%d images has no label'%len(files))
    for file in files:
        num = re.findall(r'\d+',file)[-1]
        move(file,args.inputDir + 'train/frame'+num+'.jpg')

if __name__ == '__main__':
    main()