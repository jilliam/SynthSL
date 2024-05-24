
import OpenEXR
import Imath
import array
import numpy as np
import csv
import time
import datetime
import h5py
import matplotlib.pyplot as plt
import cv2
import os
import argparse

#	pip uninstall openexr
#	pip install git+https://github.com/jamesbowman/openexrpython.git


def exr2numpy(exr, width, height, maxvalue=1.,normalize=True):
    """ converts 1-channel exr-data to 2D numpy arrays """                                                                    
    f = OpenEXR.InputFile(exr)

    # Compute the size
    dw = f.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    # (R) = [array.array('f', f.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]

    img=np.transpose(np.reshape([array.array('f', f.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")], (3, width, height)), (1, 2, 0))

    return img

def main(args):
    # frame_idx=0
    for path, subdirs, files in os.walk(args.exr_folder):
        sorted_files=sorted(files)
        for frame_idx, exr_image in enumerate(sorted_files):
            depth_data = exr2numpy(os.path.join(args.exr_folder,exr_image),img_shape[1],img_shape[0], maxvalue=15, normalize=False)
            depth_data=depth_data*255
            frame_prefix = '{:08}'.format(frame_idx)
            cv2.imwrite(os.path.join(args.out,'{}.png'.format(frame_prefix)), depth_data)
            os.remove(os.path.join(args.exr_folder,exr_image))
            # frame_idx+=1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exr_folder', type=str,
                        help='input folder, containing exr images')

    parser.add_argument('--out', type=str,
                        help='output folder, containing normal map images')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parent_path=os.path.abspath(os.path.join(args.exr_folder, os.pardir))
    listing = os.listdir(os.path.join(parent_path, 'rgb'))
    img_shape=cv2.imread(os.path.join(parent_path, 'rgb', listing[0])).shape
    # print(img_shape)
    main(args)


