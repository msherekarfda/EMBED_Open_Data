# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:24:04 2023

Simple script that walks through the dirs,
counts the dcm files and logs it

@author: RXS572
"""
import os
# import numpy as np
# import pandas as pd
# import sklearn
# import pydicom
from pydicom.filereader import read_dicomdir
import glob

# # PARAMS
base_dir = 'Z:/EXPORT/ExportBin'
out_log_file = 'C:/Users/RXS572/Documents/OUT/data_summary/data_summary2.log'
level_to_search = 1 # root + 1
# #
# # Example image reading
# img = pydicom.dcmread('Z:/EXPORT/ExportBin/0008be8e00dfd827/31869765a0fa43b38244cf9a2a5fa8bd/0b4f0c4e38d1438588ad061e39ba01d5.dcm')
# print(img)
#
patient_dirs = [f.path for f in os.scandir(base_dir) if f.is_dir()]
print('There are {} dirs at the root level'.format(len(patient_dirs)))


def findDirWithFileInLevel(path, ext, level=1):
    c = path.count(os.sep)
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(ext) and root.count(os.sep) - c - 1 <= level:
                yield root
                break

print('Scanning all dir...')
accu = []
unq_count = 0
with open(out_log_file, 'w') as fp:
    for i in findDirWithFileInLevel(base_dir, ".dcm", level_to_search):
        accu += [i]
        num_files = len(glob.glob1(i,"*.dcm"))
        unq_count += 1
        fp.write(str(unq_count) + '\t' + i + '\t' + str(num_files) + '\n')
        if unq_count % 500 == 0:
            fp.flush()
            print('Read {} so far ....'.format(len(accu)), flush=True)
        # # debug
        # if unq_count % 500 == 0:
            # break
print('There are a total of {} dirs + sub-dirs'.format(len(accu)), end="\r")
