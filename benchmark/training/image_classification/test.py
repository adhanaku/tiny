'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

test.py: performances on cifar10 test set
target performances: https://github.com/SiliconLabs/platform_ml_models/tree/master/eembc/CIFAR10_ResNetv1
'''

import numpy as np
import train
import eval_functions_eembc

import subprocess
from kaldiio import WriteHelper

TEST_SIZE = 10000

if __name__ == "__main__":

    cifar_10_dir = r'.\cifar-10-python\cifar-10-batches-py' 
    
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)
    
    
    print("Test data :", test_data.shape)
    print("Train data :",train_data.shape)
    
    # Use as ref data for POT
    #fspec = "ark:"+"input_train.ark"
    #with WriteHelper(fspec) as writer:
    #    for i in range(50000):
    #        uttname = "utt"+str(i)
    #        data = train_data[i].reshape(1, -1).astype(float)
    #        data = np.single(data)
    #        writer(uttname, data)


    fspec = "ark:"+"input_test.ark"
    with WriteHelper(fspec) as writer:
        for i in range(TEST_SIZE):
            uttname = "utt"+str(i)
            data = test_data[i].reshape(1, -1).astype(float)
            data = np.single(data)
            writer(uttname, data)
    
    print("Running GNA inference")
    
    command = ["gnat", "infer", "--model pretrainedResnet.xml", "--input input_test.ark", "-d GNA_SW_EXACT", "-o out.npz"]

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
    label_classes = np.argmax(test_labels[:TEST_SIZE],axis=1)

    print("calculate_accuracy method")
    accuracy_eembc = eval_functions_eembc.calculate_accuracy(act_results, label_classes)
    print("---------------------")
