import argparse
import sys

from utils.classification import classifier

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
    parser.add_argument("--optim", dest="optim", type=str, default="rmsprop")
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

    # Cropping Style - Ann
    parser.add_argument("--crop", dest="cropping_style", type=str, default='default')
    parser.add_argument("--xSize", dest="image_size", type=int, default=224)
    parser.add_argument("--nChannels", dest="num_channels", type=int, default=1)

    # Parameters/Hyper-parameteres for DenseNet
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--bc_mode", type=int, default=0)
    # reduction Theta at transition layer for DenseNets-BC models
    parser.add_argument("--reduction", type=float, default=1.0)
    parser.add_argument("--total_blocks", type=int, default=3)
    parser.add_argument("--growth_rate", type=int, default=12)
    parser.add_argument("--momentum", dest="momentum", type=float, default=0.9)


    # Creating Object

    opts = parser.parse_args(args[1:])
    CNN_obj = classifier(opts)
    CNN_obj.train_model() #Train/Validate the Model
    CNN_obj.test_model() #Test the Model.
    CNN_obj.do_inference() #Do inference on inference set.

    # We're done.
    return 0


if __name__ == '__main__':
    main(sys.argv)
