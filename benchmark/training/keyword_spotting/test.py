#!/usr/bin/env python

import numpy as np
import os

import get_dataset as kws_data
import kws_util
import eval_functions_eembc

import tensorflow_datasets as tfds
import subprocess
from kaldiio import WriteHelper
import tensorflow


num_classes = 12 
FLAGS = None

if __name__ == '__main__':
  
  Flags, unparsed = kws_util.parse_command()

  print('We will download data to {:}'.format(Flags.data_dir))

  ds_train, ds_test = kws_data.get_training_data(Flags)
  print("Done getting data")
   
  # Get the number of elements in the dataset
  num_elements = tensorflow.data.experimental.cardinality(ds_test).numpy()      
  #num_elements_train = tensorflow.data.experimental.cardinality(ds_train).numpy()      
  print("Number of elements in test dataset:", num_elements) 

  
  # Initialize an empty NumPy array to store the data
  data_array = np.empty((num_elements,) + ds_test.element_spec[0].shape, dtype=np.float32)
  #train_data_array = np.empty((num_elements_train,) + ds_train.element_spec[0].shape, dtype=np.float32)

  # Iterate over the dataset and fill the data array
  for i, (inputs, _) in enumerate(ds_test):
    data_array[i] = inputs.numpy()

  print("Shape of the data array:", data_array.shape)

  # Use as ref data for POT
  #Fill array with train data 
  #for i, (inputs, _) in enumerate(ds_train):
  #  train_data_array[i] = inputs.numpy()

  #print("Shape of the train data array:", train_data_array.shape)
  
  #fspec = "ark:"+"input_train.ark"
  #with WriteHelper(fspec) as writer:
  #    for i in range(num_elements_train):
  #        uttname = "utt"+str(i)
  #        data =train_data_array[i].reshape(1, -1)
  #        data = np.single(data)
  #        writer(uttname, data)
        
  
  
  # Initialize an empty NumPy array to store the labels
  labels_array = np.empty(num_elements, dtype=np.int64)

  # Iterate over the dataset and fill the labels array
  for i, (_, labels) in enumerate(ds_test):
    labels_array[i] = labels.numpy()
    
  print("Shape of the labels array:", labels_array.shape)
  
  fspec = "ark:"+"input.ark"
  with WriteHelper(fspec) as writer:
      for i in range(num_elements):
          uttname = "utt"+str(i)
          data =data_array[i].reshape(1, -1)
          data = np.single(data)
          writer(uttname, data)
  
  print("Running GNA inference")      
  command = ["gnat", "infer", "--model kws_ref_model_float32.xml", "--input input.ark", "-d GNA_SW_EXACT", "-o out.npz"]

  try:
      output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
  except subprocess.CalledProcessError as e:
      print("Error running command:", e.output)
    
  print("Inference Complete")
  gna_out = np.load('out.npz')
  argmax_results = []  

  for key in gna_out.keys():
      argmax_results.append(np.argmax(gna_out[key], axis=1))
    
  act_results = np.concatenate(argmax_results)   

  print("Calculating accuracy")
  accuracy_eembc = eval_functions_eembc.calculate_accuracy(act_results, labels_array)
  print("---------------------")
