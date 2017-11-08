import argparse
import sys,os
import numpy as np

from utils.classification import classifier
from data_pipeline.create_cv_h5 import create_val_split

def main(args):
    """
    Main function to parse arguments.
    INPUTS:
    - args: (list of strings) command line arguments
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Do CNN Segmentation.")

    # Paths: arguments for filepath to misc.
    parser.add_argument("--pTrain", dest="path_train", type=str, default=None)
    parser.add_argument("--pVal", dest="path_validation", type=str, default=None)
    parser.add_argument("--pTest", dest="path_test", type=str, default=None)
    parser.add_argument("--pInf", dest="path_inference", type=str, default=None)
    parser.add_argument("--pModel", dest="path_model", type=str, default=None)
    parser.add_argument("--pLog", dest="path_log", type=str, default=None)
    parser.add_argument("--pVis", dest="path_visualization", type=str, default=None)

    # Experiment Specific Parameters (i.e. architecture)
    parser.add_argument("--name", dest="name", type=str, default="noname")
    parser.add_argument("--net", dest="network", type=str, default="GoogLe")
    parser.add_argument("--nClass", dest="num_class", type=int, default=2)
    parser.add_argument("--nGPU", dest="num_gpu", type=int, default=1)
    
    # Hyperparameters
    parser.add_argument("--lr", dest="lr", type=float, default=0.001)
    parser.add_argument("--dec", dest="lr_decay", type=float, default=1.0)
    parser.add_argument("--do", dest="keep_prob", type=float, default=0.5)
    parser.add_argument("--l2", dest="l2", type=float, default=0.0000001)
    parser.add_argument("--l1", dest="l1", type=float, default=0.0)
    parser.add_argument("--bs", dest="batch_size", type=int, default=12)
    parser.add_argument("--ep", dest="max_epoch", type=int, default=10)
    parser.add_argument("--time", dest="max_time", type=int, default=1440)

    # Switches
    parser.add_argument("--bLo", dest="bool_load", type=int, default=0)
    parser.add_argument("--bDisp", dest="bool_display", type=int, default=1)
    parser.add_argument("--bConf", dest="bool_confusion", type=int, default=0)
    parser.add_argument("--bKappa", dest="bool_kappa", type=int, default=0)
    parser.add_argument("--rSeed", dest="rand_seeds", type=str, default=1)
    parser.add_argument("--mlo", dest="mlo_only", type=int, default=0)

    # Creating Object
    opts = parser.parse_args(args[1:])
    acc_train = []
    acc_val = []
    acc_test = []
    seeds_str = opts.rand_seeds.split(',')
    seeds= [int(a) for a in seeds_str]
    
    for ii,rs in enumerate(seeds):
        print "CREATING CV SPLIT %d of %d" % (ii, len(seeds))
        #Can set suffix argument here using argparse
        if opts.mlo_only:
            suf = '_mlo'
        else:
            suf = ''
        create_val_split(rs,suf,opts.mlo_only)
        print "TRAINING MODEL %d of %d" % (ii, len(seeds))
        CNN_obj = classifier(opts)
        CNN_obj.train_model() #Train/Validate the Model
        acc_tr_cv, acc_val_cv, acc_test_cv = CNN_obj.test_model() #Test the Model.
        CNN_obj.do_inference() #Do inference on inference set.
        acc_tr.append(acc_tr_cv)
        acc_val.append(acc_val_cv)
        acc_test.append(acc_val_cv) 
    
    #Printing Accuracies
    acc_tr = np.array(acc_tr)
    acc_val = np.array(acc_val)
    acc_test = np.array(acc_test)
    print"Mean Train Accuracy: %.2f \n Mean Validation Accuracy: %.2f \n Mean Test Accuracy: %.2f \n"\
          % (np.mean(acc_tr), np.mean(acc_val), np.mean(acc_test))
    print"Train Accuracy Std: %.2f \n Validation Accuracy Std: %.2f \n Test Accuracy Std: %.2f \n"\
          % (np.mean(acc_tr), np.std(acc_val), np.std(acc_test))
    
    #Saving accuracies
    print "Saving Accuracies..."
    logpath = os.path.dirname(CNN_obj.path_log)
    outfile = os.path.join(logpath,'run_accuracies.npz')      
    np.savez(outfile, acc_test=acc_test,acc_val=acc_val,acc_tr=acc_tr)
    print "Program Complete!"
    # We're done.
    return 0
    

if __name__ == '__main__':
    main(sys.argv)
