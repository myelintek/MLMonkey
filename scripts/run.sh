#!/bin/bash
for ARGUMENT in "$@"
do
  KEY=$(echo $ARGUMENT | cut -f1 -d=)
  VALUE=$(echo $ARGUMENT | cut -f2 -d=)
  case "$KEY" in
    rnn_translator_bs)     rnn_translator_bs=${VALUE} ;;
    object_detection_bs)   image_segm_bs=${VALUE} ;;
    image_cl_bs)           image_cl_bs=${VALUE} ;;
    num_gpus)              num_gpus=${VALUE} ;;
    debug)                 debug=${VALUE} ;;
    quick)                 quick=${VALUE} ;;
    *)
  esac
done

total_gpus=`nvidia-smi -L | wc -l`
echo "total # of gpu: $total_gpus"

# default value
num_gpus=${num_gpus:-$total_gpus}
debug=${debug:-false}
quick=${quick:-false}

num_batches=2500
# debug mode
if ${debug} ; then
  echo "!!!debug mode!!!"
  echo "hardcode num_batches to 100 in debug mode"
  num_batches=100
fi

rnn_translator_bs=${rnn_translator_bs:-128}
object_detection_bs=${object_detection_bs:-128} 
image_classification_bs=${image_classification_bs:-1664}
num_gpus=${num_gpus:-8} 

mkdir -p /tmp/logs
cd /workspace
mkdir -p logs
cd logs
mkdir -p gpu_scalability fp16 full_imagenet
cd /run_benchmarks/benchmarks/scripts/tf_cnn_benchmarks

LOG_RDIR=/workspace/logs
TFB_DIR=$LOG_RDIR/tf_benchmarks
#RTR_DIR=$LOG_RDIR/rnn_translator
#OBD_DIR=$LOG_RDIR/object_detection

RESNET_COMMON="--data_format=NCHW --model=resnet50 --optimizer=momentum \
 --variable_update=replicated --all_reduce_spec=nccl \
 --gradient_repacking=8 --use_fp16"


find_max_batch_size()
{
  # debug mode
  if ${debug} ; then
    echo "hardcode batch size to 128 in debug mode."
    bsfp16=128
    return
  fi
  mkdir -p /tmp/logs
  bs1gpu=( 1024 512 256 128 64 32 16 8 4 2 )
  echo "Running Imagenet with Resnet50 benchmark"
  echo "Checking maximum possible batch size for fp16 with synthetic data"
  for bsize in ${bs1gpu[@]}
  do
    python -u tf_cnn_benchmarks.py \
        ${RESNET_COMMON} \
        --num_gpus=1 \
        --batch_size=$bsize \
        --num_batches=10 \
        --gpu_memory_frac_for_testing=0.8 2>&1 | tee /tmp/logs/find_max_bs_fp16.log | \
        pv --eta --line-mode --name " Test Batch Size $bsize" -b -p --timer -s 166 > /dev/null
    test ${PIPESTATUS[0]} -eq 0 && bsfp16=$bsize && echo "Maximum batch size for fp16 with synthetic data: $bsize" && break
  done
}

quick_test()
{
  if ${quick} ; then
    echo "start temperature stress test"
    ( time python tf_cnn_benchmarks.py \
          ${RESNET_COMMON} \
          --num_gpus=$num_gpus \
          --batch_size=$bsfp16 \
          --num_epochs=10 ) 2>&1
    exit
  fi
}

export TIMEFORMAT='%E real,  %U user,  %S sys'

get_gpus()
{
  n=$num_gpus
  n_gpus=($num_gpus)
  while (( $n>1 )); do
    n=$(( $n / 2 ))
    n_gpus=("${n_gpus[@]}" $n)     
  done
}

gpus_scalability_test()
{
  mkdir -p $TFB_DIR/gpu_scalability
  get_gpus
  echo "Run gpu scalability test for ${n_gpus[@]} GPUs"
  for gpus in ${n_gpus[@]}
  do
    echo "Running gpu scalability test, output is redirected to $TFB_DIR/gpu_scalability/$gpus.log"
    ( time python -u tf_cnn_benchmarks.py \
        ${RESNET_COMMON} \
        --num_gpus=$gpus \
        --batch_size=$bsfp16 \
        --num_batches=${num_batches} ) 2>&1 | tee $TFB_DIR/gpu_scalability/$gpus.log | \
        pv --eta --line-mode --name " Scalability Test $gpus GPU(s)" -b -p --timer -s 415 > /dev/null
  done
}


real_vs_synthetic_data()
{
  mkdir -p $TFB_DIR/fp16
  echo "Running real imagenet test, logs are redirected to $TFB_DIR/fp16/imagenet.log"
  #real
  if [ -d "/tfrecords" ] ; then #if real data is not mounted skip the test
    ( time python -u tf_cnn_benchmarks.py \
         ${RESNET_COMMON} \
         --num_gpus=$num_gpus \
         --batch_size=$bsfp16 \
         --num_batches=${num_batches} \
         --data_dir=/tfrecords \
         --data_name=imagenet ) 2>&1 | tee $TFB_DIR/fp16/imagenet.log | \
         pv --eta --line-mode --name " ResNet50 with ImageNet dataset" -b -p --timer -s 415 > /dev/null
    return_code=${PIPESTATUS[0]}
    #if there was an error, probably due to OOM, try reducing batch size
    if [ $return_code != 0 ]  ; then
      bsfp16=$(($bsfp16 / 2))
      echo "Reduce batch size for real data: $bsfp16"
      ( time python -u tf_cnn_benchmarks.py \
            ${RESNET_COMMON} \
            --num_gpus=$num_gpus \
            --batch_size=$bsfp16 \
            --num_batches=${num_batches} \
            --data_dir=/tfrecords \
            --data_name=imagenet ) 2>&1 | tee $TFB_DIR/fp16/imagenet.log | \
         pv --eta --line-mode --name " ResNet50 with ImageNet dataset" -b -p --timer -s 415 > /dev/null
    fi
  else
    echo "ERROR: /tfrecords folder is not found" > $TFB_DIR/fp16/imagenet.log
  fi
  #synthetic 
  echo "Running synthetic data test, logs are redirected to $TFB_DIR/fp16/synthetic.log"
  ( time python tf_cnn_benchmarks.py \
        ${RESNET_COMMON} \
        --num_gpus=$num_gpus \
        --batch_size=$bsfp16 \
        --num_batches=${num_batches} \
        --data_name=imagenet ) 2>&1 | tee $TFB_DIR/fp16/synthetic.log | \
        pv --eta --line-mode --name " ResNet50 with Synthetic dataset" -b -p --timer -s 415 > /dev/null
}


full_imagenet()
{
  # debug mode
  if ${debug} ; then
    TRAIN_STEPS="--num_batches=100"
    VAL_STEPS="--num_batches=100"
  else
    TRAIN_STEPS="--num_epochs=90"
    VAL_STEPS="--num_epochs=1"
  fi
  mkdir -p $TFB_DIR/full_imagenet
  #full imagenet training to 90 epoch with maximum batch size
  echo "Running full imagenet training, $TFB_DIR/full_imagenet/train_ep90_bs$bsfp16.log"
  #scale lr according to batch size: batch_size=256, lr=0.1
  lr1=$(echo "scale=6; $bsfp16 / 256 * $num_gpus / 10" | bc )
  lr2=$(echo "scale=6; $lr1/10" | bc )
  lr3=$(echo "scale=6; $lr2/10" | bc )
  lr4=$(echo "scale=6; $lr3/10" | bc ) 
  #train
  [ -d "/workspace/resnet50_train_full" ] && rm -rf /workspace/resnet50_train_full
  ( time python tf_cnn_benchmarks.py \
        ${RESNET_COMMON} \
        --num_gpus=$num_gpus \
        --batch_size=$bsfp16 \
        ${TRAIN_STEPS} \
        --weight_decay=4e-5 \
        --data_dir=/tfrecords/ \
        --data_name=imagenet \
        --print_training_accuracy=True \
        --train_dir=/workspace/resnet50_train_full \
        --num_learning_rate_warmup_epochs=5 \
        --piecewise_learning_rate_schedule="$lr1;30;$lr2;60;$lr3;80;$lr4" ) 2>&1 | tee $TFB_DIR/full_imagenet/train_ep90_bs$bsfp16.log | \
        pv --eta --line-mode --name " ResNet with ImageNet dataset for 90 epochs" -b -p --timer -s 45000 > /dev/null
  #eval
  ( time python tf_cnn_benchmarks.py \
        ${RESNET_COMMON} \
        --num_gpus=$num_gpus \
        --batch_size=$bsfp16 \
        ${VAL_STEPS} \
        --data_dir=/tfrecords/ \
        --data_name=imagenet \
        --train_dir=/workspace/resnet50_train_full \
        --num_learning_rate_warmup_epochs=5 \
        --piecewise_learning_rate_schedule="$lr1;30;$lr2;60;$lr3;80;$lr4" \
        --eval ) 2>&1 | tee $TFB_DIR/full_imagenet/val_ep90_bs$bsfp16.log | \
        pv --eta --line-mode --name " ResNet evaluation" -b -p --timer -s 800 > /dev/null
}

#nvidia-optimized mlperf, default parameters
rnn_translator()
{
  cd /run_benchmarks/results/v0.5.0/nvidia/submission/code/rnn_translator  
  if [ ! -d "/workspace/datasets/rnn_translator" ] ; then
    mkdir -p /workspace/datasets/rnn_translator
    bash download_dataset.sh /workspace/datasets/rnn_translator
  fi
  docker pull 140.96.29.39:5000/myelintek/mlperf-nvidia:rnn_translator
  docker tag 140.96.29.39:5000/myelintek/mlperf-nvidia:rnn_translator mlperf-nvidia:rnn_translator
  cd pytorch
  sed -i "s/BATCH=[[:digit:]]*/BATCH=$rnn_translator_bs/g"  config_DGX1.sh
  sed -i "s/TEST_BATCH_SIZE=[[:digit:]]*/TEST_BATCH_SIZE=$rnn_translator_bs/g"  config_DGX1.sh
  sed -i "s/DGXNGPU=[[:digit:]]*/DGXNGPU=$num_gpus/g"  config_DGX1.sh
  DATADIR=$WORKDIR/datasets/rnn_translator LOGDIR=/workspace/logs/rnn_translator DGXSYSTEM=DGX1 ./run.sub
}

object_detection()
{
  cd /run_benchmarks/results/v0.5.0/nvidia/submission/code/object_detection
   if [ ! -d "/workspace/datasets/coco" ] ; then
     ./download_dataset.sh
   fi
  cd pytorch
  docker pull 140.96.29.39:5000/myelintek/mlperf-nvidia:object_detection
  docker tag 140.96.29.39:5000/myelintek/mlperf-nvidia:object_detection mlperf-nvidia:object_detection
#  sed -i "s/BATCH=[[:digit:]]*/BATCH=$object_detection_bs/g"  config_DGX1.sh
  sed -i "s/DGXNGPU=[[:digit:]]*/DGXNGPU=$num_gpus/g"  config_DGX1.sh
  DATADIR=$WORKDIR/datasets/coco LOGDIR=/workspace/logs/object_detection  ./run.sub  
}

image_classification()
{
  cd /run_benchmarks/results/v0.5.0/nvidia/submission/code/image_classification/mxnet/
  docker pull 140.96.29.39:5000/myelintek/mlperf-nvidia:image_classification
  docker tag 140.96.29.39:5000/myelintek/mlperf-nvidia:image_classification mlperf-nvidia:image_classification
  sed -i "s/BATCHSIZE=[[:digit:]]*/BATCHSIZE=$image_classification_bs/g"  config_DGX1.sh
  sed -i "s/DGXNGPU=[[:digit:]]*/DGXNGPU=$num_gpus/g"  config_DGX1.sh
  DATADIR=$WORKDIR/datasets/imagenet-mxnet/rec LOGDIR=/workspace/logs/image_classification  ./run.sub
}

get_cuda_p2p()
{
  cd /workspace
  git clone https://github.com/NVIDIA/cuda-samples.git
  cd cuda-samples/
  git fetch && git fetch --tags
  git checkout v10.0 #TODO sync CUDA version with mlmonkey image
  cd Samples/p2pBandwidthLatencyTest/
  make
  mkdir -p /workspace/logs/hw_info
  ./p2pBandwidthLatencyTest &> /workspace/logs/hw_info/p2p.log
}

hwinfo()
{
  mkdir -p /workspace/logs/hw_info
  nvidia-smi &> /workspace/logs/hw_info/nvidia-smi.log
  lshw -html &> /workspace/logs/hw_info/lshw.html
}

find_max_batch_size
quick_test
gpus_scalability_test
real_vs_synthetic_data
sleep 60s
full_imagenet
sleep 60s
rnn_translator
sleep 60s
object_detection
# image_classification
sleep 10s
hwinfo
get_cuda_p2p
