
import os
import glob
import argparse
import re
from shutil import copyfile,move
import xml.etree.ElementTree as ET


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="generate gtround truth .txt")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .xml files are stored",
                        type=str)
    parser.add_argument("-s",
                        "--set_num",
                        help="set number.",
                        type=int)
    args = parser.parse_args()

    if(args.inputDir is None or args.set_num is None):
        return
    
    assert(os.path.isdir(args.inputDir))
    
    #-----------------------------------------------------------
    #-----------------------------------------------------------
    files = glob.glob(args.inputDir + '*.xml')
    files.sort()
    for i,xml_file in enumerate(files):
        file_name = os.path.splitext(os.path.basename(xml_file))
        file = open('eval_box/set%d/gt/'%args.set_num+file_name[0]+'.txt','w')
        print(file_name[0])
        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()
        bndbox = root.find('object/bndbox')
        if bndbox != None:
            name = root.find('object/name').text
            xmin = bndbox.find('xmin').text
            xmax = bndbox.find('xmax').text
            ymin = bndbox.find('ymin').text
            ymax = bndbox.find('ymax').text
            print(str(i)+' '+name+' '+xmin+' '+ymin+' '+xmax+' '+ymax+'\n')
            file.write(name+' '+xmin+' '+ymin+' '+xmax+' '+ymax+'\n')

        file.close()

            
if __name__ == "__main__":
    main()
