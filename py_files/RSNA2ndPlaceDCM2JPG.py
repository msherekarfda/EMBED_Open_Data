import cv2
#import gdcm
import dicomsdl
import pydicom
from pydicom.filebase import DicomBytesIO
import os
import glob
import shutil
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import pandas as pd

import torch
import torch.nn.functional as F

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.types import DALIDataType
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type


def convert_dicom_to_j2k(fe, save_folder="", DATA=""):
    # patient = fe.split('/')[-2]
    # image = fe.split('/')[-1][:-4]
    if DATA == 'RSNA' or 'CBIS':
        patient = fe.split('/')[-2]
        image = fe.split('/')[-1][:-4]
    elif DATA == 'EMBED':
        cohorts = {"cohort_1/", "cohort_2/"}
        fe2 = fe
        for d in cohorts:
            fe2 = fe2.replace(d, '-')
        print('fe2 convert', fe2)
        patient_image = fe2.split('-')[-1].split(os.sep)
        patient = patient_image[0]
        image = "_".join(patient_image[1:]).split('.dcm')[0]
        print('patient convert', patient)
        print('image convert', image)
    else:
        print('*ERROR. UNKNOWN DATA TYPE. NOTHING TO DO. EXITING')
        return
    dcmfile = pydicom.dcmread(fe)

    if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        with open(fe, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  #<---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(save_folder + f"{patient}_{image}.jp2", "wb") as binary_file:
            binary_file.write(hackedbitstream)

            
@pipeline_def
def j2k_decode_pipeline(j2kfiles):
    jpegs, _ = fn.readers.file(files=j2kfiles)
    images = fn.experimental.decoders.image(jpegs, device='mixed', output_type=types.ANY_DATA, dtype=DALIDataType.UINT16)
    return images


def dicomsdl_to_numpy_image(dicom, index=0):
    info = dicom.getPixelDataInfo()
    dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    outarr = np.empty(shape, dtype=dtype)
    dicom.copyFrameData(index, outarr)
    return outarr


def load_img_dicomsdl(f):
    return dicomsdl_to_numpy_image(dicomsdl.open(f))


def process(f, size=512, save_folder="", DATA=""):
    # patient = f.split('/')[-2]
    # image = f.split('/')[-1][:-4]
    if DATA == 'RSNA':
        patient = f.split('/')[-2]
        image = f.split('/')[-1][:-4]
    elif DATA == 'EMBED':
        cohorts = {"cohort_1/", "cohort_2/"}
        fe2 = f
        for d in cohorts:
            fe2 = fe2.replace(d, '-')
        print('fe2 process', fe2)
        patient_image = fe2.split('-')[-1].split(os.sep)
        patient = patient_image[0]
        image = "_".join(patient_image[1:]).split('.dcm')[0]
        print('patient process', patient)
        print('image process', image)
    else:
        print('*ERROR. UNKNOWN DATA TYPE. NOTHING TO DO. EXITING')
        return

    dicom = pydicom.dcmread(f)

    if dicom.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':  # ALREADY PROCESSED
        return

    try:
        img = load_img_dicomsdl(f)
    except:
        img = dicom.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cv2.resize(img, (size, size),interpolation = cv2.INTER_CUBIC)

    cv2.imwrite(save_folder + f"{patient}_{image}.png", (img * 255).astype(np.uint8))


def dcm_to_jpg(args, IMG_PATH):
    # #
    ### Process jpeg compressed dicoms on GPU
    # - Convert files to j2k
    # - Load j2k files, resize & scale on GPU !
    # - Processing is done per batch not to run out of disk space    
    
    # # get cases to convert from the input list file
    df1 = pd.read_csv(args.train_csv_pth)
    df2 = pd.read_csv(args.valid_csv_pth)
    df = pd.concat([df1, df2], ignore_index=True)

    if args.data == 'RSNA':
        test_images = [os.path.join(args.dcm_root_dir_pth, str(p), str(v)) + '.dcm' for p, v in zip(df['patient_id'], df['image_id'])]
    elif args.data == 'EMBED':
        test_images = [os.path.join(args.dcm_root_dir_pth, str(p)) + '.dcm' for p in df['image_id']]
    else:
        print('=ERROR. UNKNOWN DATA TYPE. NOTHING TO DO. EXITING')
        return
    print('test_images', test_images)
    print("Number of images :", len(test_images))
    
    SIZE = 1536
    os.makedirs(args.save_img_root_dir_pth, exist_ok=True)
    # #
    if len(test_images) > 100:
        N_CHUNKS = 4
    else:
        N_CHUNKS = 1
    CHUNKS = [(len(test_images) / N_CHUNKS * k, len(test_images) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
    CHUNKS = np.array(CHUNKS).astype(int)
    # # this is a temp dir that gets deleted
    J2K_FOLDER = os.path.join(args.save_img_root_dir_pth, 'j2k') + os.sep
    
    for chunk in tqdm(CHUNKS):
        os.makedirs(J2K_FOLDER, exist_ok=True)
        try:
            _ = Parallel(n_jobs=2)(
                    delayed(convert_dicom_to_j2k)(img, save_folder=J2K_FOLDER, DATA=args.data)
                    for img in test_images[chunk[0]: chunk[1]]
                )
        except: 
            print("Error occurred during J2K conversion. Continuing to the next chunk.")
            continue
            
        j2kfiles = glob.glob(J2K_FOLDER + "*.jp2")

        if not len(j2kfiles):
            continue

        pipe = j2k_decode_pipeline(j2kfiles, batch_size=1, num_threads=2, device_id=0, debug=True)
        pipe.build()

        for i, f in enumerate(j2kfiles):
            try:
                if args.data == 'RSNA':
                    patient, image = f.split(os.sep)[-1][:-4].split('_')
                elif args.data == 'EMBED':
                    print('+ERROR. NOT IMPLEMENTED. NOTHING TO DO. EXITING')
                    return
                else:
                    print('+ERROR. UNKNOWN DATA TYPE. NOTHING TO DO. EXITING')
                    return
                dicom = pydicom.dcmread(IMG_PATH + f"{patient}/{image}.dcm")

                out = pipe.run()

                # Dali -> Torch
                img = out[0][0]
                img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
                feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
                img = img_torch.float()

                # Scale, resize, invert on GPU !
                min_, max_ = img.min(), img.max()
                img = (img - min_) / (max_ - min_)
                if SIZE:
                    img = F.interpolate(img.view(1, 1, img.size(0), img.size(1)), (SIZE, SIZE), mode="bicubic")[0, 0]
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    img = 1 - img
                # Back to CPU + SAVE
                img = (img * 255).cpu().numpy().astype(np.uint8)
                cv2.imwrite(args.save_img_root_dir_pth + f"{patient}_{image}.png", img)
            except Exception as e:
                print(f"Error processing image {f}: {str(e)}")
                continue
        shutil.rmtree(J2K_FOLDER)
    
    # # Process the rest on CPU
    print('Process the rest on CPU')
    _ = Parallel(n_jobs=2)(
                    delayed(process)(img, size=SIZE, save_folder=args.save_img_root_dir_pth, DATA=args.data)
                    for img in tqdm(test_images) 
                )
