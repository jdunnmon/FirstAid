EXEC_SCRIPT=../train_CNNclassification.py

#TRAIN_PATH_ROOT="/Users/annhe/Projects/tandaExperiment/ddsm-h5-750/"
#TRAIN_PATH_ROOT="/scratch/users/annhe/ddsm-h5-750/"
TRAIN_PATH_ROOT="/mnt/data-1/users/jdunnmon/data_aug/ddsm-data/ann_preproc/h5_data_all/"
DATA_SUF="_all"
PATH_TRAIN="$TRAIN_PATH_ROOT/h5_train_set$DATA_SUF"
PATH_VAL="$TRAIN_PATH_ROOT/h5_val_set$DATA_SUF"
PATH_TEST="$TRAIN_PATH_ROOT/h5_test_set$DATA_SUF"
OUTPUT_PATH="/mnt/data-1/users/jdunnmon/data_aug/firstaid/all_runs/logs/ann_densenet_test"

export CUDA_VISIBLE_DEVICES=0

NET_NAME=Dense
EPOCHS=200

EXP_NAME=hp_search_${NET_NAME}_ann_run_v3
START_DATE=`date +"%m_%d_%y"`
crop='random'
xSize=224
nChannels=1

# removed yellowfin from opt
for opt in sgd
do
for lr in  0.1 0.01 0.001  
do
for dp in  0.9
do
for l2 in 0
do
for dec in 1.0 0.99
do
for l1 in 0
do
for bs in 96 64 32 16
do
  echo "Running Case with OPT = $opt, LR = $lr, L2= $l2, DO = $dp, DEC= $dec, l1=$l1, BS=$bs"
  TRIAL_NAME=${EXP_NAME}_ep_${EPOCHS}_opt_${opt}_lr_${lr}_dp_${dp}_l2_${l2}_dec_${dec}_l1_${l1}_bs_${bs}
  TIME=`date +"%H_%M_%S"`
  LOGDIR="$OUTPUT_PATH/${START_DATE}/${EXP_NAME}/${TRIAL_NAME}"
  mkdir -p $LOGDIR
  PATH_SAVE="$LOGDIR/model"
  PATH_VIS="$LOGDIR/vis.png"
  LOGFILE="$LOGDIR/terminal_run_log_${TIME}.log"
  PATH_LOG="$LOGDIR/internal_run_log_${TIME}.log"
  echo "Saving log to '$LOGFILE'"

  #source set_env.sh
  python $EXEC_SCRIPT --pTrain $PATH_TRAIN --pVal $PATH_VAL --pTest $PATH_TEST --pModel $PATH_SAVE --pVis $PATH_VIS --pLog $PATH_LOG --name $EXP_NAME --net $NET_NAME --nClass 2 --nGPU 1 --lr $lr --dec $dec --do $dp --l2 $l2 --l1 $l1 --bs $bs --ep $EPOCHS --optim $opt --crop $crop --xSize $xSize --nChannels $nChannels --time 1440 --bLo 0 --bDisp 0 2>&1 | tee $LOGFILE
 done
 done
 done
 done
 done
 done
 done

SCRIPT_PATH=${0}
SCRIPT_OUT_PATH="$OUTPUT_PATH/${START_DATE}/${EXP_NAME}"
cp -r $SCRIPT_PATH $SCRIPT_OUT_PATH
echo "Run complete. This script copied to $SCRIPT_OUT_PATH"
