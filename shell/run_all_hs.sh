EXEC_SCRIPT=/lfs/local/0/jdunnmon/repos/FirstAid/train_CNNclassification.py
TRAIN_PATH_ROOT="/lfs/local/0/jdunnmon/data_aug/ddsm-data/ann_preproc/h5_data_all/"
DATA_SUF="_all"
PATH_TRAIN="$TRAIN_PATH_ROOT/h5_train_set$DATA_SUF"
PATH_VAL="$TRAIN_PATH_ROOT/h5_val_set$DATA_SUF"
PATH_TEST="$TRAIN_PATH_ROOT/h5_test_set$DATA_SUF"
OUTPUT_PATH="/lfs/local/0/jdunnmon/data_aug/firstaid/all_runs/logs"

CUDA_VISIBLE_DEVICES=1

NET_NAME=VGG19
EPOCHS=200

EXP_NAME=hp_search_${NET_NAME}
START_DATE=`date +"%m_%d_%y"`

for opt in adam rmsprop 
do
for lr in 0.01 0.001 0.0001 
do
for dp in 0.9 1.0
do
for l2 in 0.01 0.0001 0.000001
do
for dec in 0.99 0.95
do
for l1 in 0
do
for bs in 64 128
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
  python $EXEC_SCRIPT --pTrain $PATH_TRAIN --pVal $PATH_VAL --pTest $PATH_TEST --pModel $PATH_SAVE --pVis $PATH_VIS --pLog $PATH_LOG --name $EXP_NAME --net $NET_NAME --nClass 2 --nGPU 1 --lr $lr --dec $dec --do $dp --l2 $l2 --l1 $l1 --bs $bs --ep $EPOCHS --optim $opt --time 1440 --bLo 0 --bDisp 0 2>&1 | tee $LOGFILE
 done
 done
 done
 done
 done
 done
 done

  
