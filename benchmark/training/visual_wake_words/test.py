import wget
import tarfile
import os, csv, PIL
import numpy as np
import tensorflow as tf
import subprocess
from kaldiio import WriteHelper
NUM_VAL_SAMPLES = 1000


url = 'https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz'
output_path = './vw_coco2014_96.tar.gz'
extract_dir = './vw_coco_extract'


wget.download(url, out=output_path)
print("\n Download complete!")

# Open the tar.gz file
with tarfile.open(output_path, 'r:gz') as tar:
    # Extract all contents to the specified directory
    tar.extractall(path=extract_dir)
print("\n Extraction complete!")

#Convert 1000 images from train set to npy array
train_array_list = []
with open('image_train.csv') as f:
  for row in csv.reader(f):
    # get image 'img' from a file listed in image_train: row[0] = filename;
    # row[2] = '1' or '0' for person/non-person; ignore row[1]
    fname = 'COCO_train2014_' + os.path.splitext(row[0])[0] + '.jpg'
    dir = 'person/' if row[2] == '1' else 'non_person/'
    img = PIL.Image.open('./vw_coco_extract/vw_coco2014_96/' + dir + fname)

    # convert to numpy array and reshape from (96, 96, 3) to (1, 96, 96, 3)
    img = np.asarray(img)

    # for float-model, convert uint8 to float32
    fp_img = (img / 256).astype(np.float32)

    train_array_list.append(fp_img)

#Use as ref data to run  POT    
fspec = "ark:"+"input_train_vww.ark"
with WriteHelper(fspec) as writer:
     for i in range(NUM_VAL_SAMPLES):
         uttname = "utt"+str(i)
         data = train_array_list[i].reshape(1, -1)
         data = np.single(data)
         writer(uttname, data)



#Convert 1000 images from val/test set to npy array
image_array_list = []
labels = []
test_cnt = 0

with open('y_labels.csv') as f:
  for row in csv.reader(f):
    # get image 'img' from a file listed in y_labels: row[0] = filename;
    # row[2] = '1' or '0' for person/non-person; ignore row[1]
    fname = 'COCO_val2014_' + os.path.splitext(row[0])[0] + '.jpg'
    dir = 'person/' if row[2] == '1' else 'non_person/'
    img = PIL.Image.open('./vw_coco_extract/vw_coco2014_96/' + dir + fname)

    # convert to numpy array and reshape from (96, 96, 3) to (1, 96, 96, 3)
    img = np.asarray(img)

    # for float-model, convert uint8 to float32
    fp_img = (img / 256).astype(np.float32) 

    image_array_list.append(fp_img)

    labels.append(row[2])
    
fspec = "ark:"+"input_test_vww.ark"
with WriteHelper(fspec) as writer:
     for i in range(NUM_VAL_SAMPLES):
         uttname = "utt"+str(i)
         data = image_array_list[i].reshape(1, -1)
         data = np.single(data)
         writer(uttname, data)

print("Running GNA inference")       
command = ["gnat", "infer", "--model vww_96_float.xml", "--input input_test_vww.ark", "-d GNA_SW_EXACT", "-o out.npz"]
#command = ["gnat", "infer", "--model vww_96_float.xml", "--input input_test_vww.ark", "-d CPU", "-o out.npz"]

try:
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
except subprocess.CalledProcessError as e:
    print("Error running command:", e.output)
    
print("Inference Complete")
gna_out = np.load('out.npz')

for key, label in zip(gna_out.keys(), labels):
     if label == '1' and gna_out[key][0][1] >= 0.5:
        test_cnt += 1
     elif label == '0' and gna_out[key][0][1] < 0.5:
        test_cnt += 1

print('Accuracy:', 100 * test_cnt / NUM_VAL_SAMPLES)
