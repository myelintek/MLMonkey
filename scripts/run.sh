#!/bin/bash
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
#    echo $KEY
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   
#    echo $VALUE
    case "$KEY" in
            rnn_translator_bs)              rnn_translator_bs=${VALUE} ;;
            object_detection_bs)                  image_segm_bs=${VALUE} ;;     
            num_gpus)                       num_gpus=${VALUE} ;;
            *)   
    esac    


done

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

find_max_batch_size()
{
  bs1gpu=( 1024 512 256 128 64 32 16 8 4 2 )
  echo "Running Imagenet with Resnet50 benchmark"
  echo "Checking maximum possible batch size for fp16 with synthetic data"
  for bsize in ${bs1gpu[@]}
  do
    python tf_cnn_benchmarks.py --num_gpus=8 --batch_size=$bsize --num_batches=10 --model=resnet50 --variable_update=replicated --use_fp16 --data_format=NCHW --optimizer=momentum --gradient_repacking=8  --all_reduce_spec=nccl &> /tmp/logs/find_max_bs_fp16.log
    test $? -eq 0 && bsfp16=$bsize && echo "Maximum batch size for fp16 with synthetic data: $bsize" && break
  done
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
#  echo ${n_gpus[@]}
}

gpus_scalability_test()
{
  get_gpus
  for gpus in ${n_gpus[@]}
  do
    echo "Running gpu scalability test, output is redirected to $TFB_DIR/gpu_scalability_fp16/$gpus.log"
    ( time python tf_cnn_benchmarks.py \
                  --num_gpus=$gpus \
                  --batch_size=$bsfp16 \
                  --num_epochs=1 \
                  --model=resnet50 \
                  --variable_update=replicated \
                  --use_fp16 \
                  --all_reduce_spec=nccl ) &> $TFB_DIR/gpu_scalability/$gpus.log
  done
}


real_vs_synthetic_data()
{
  echo "Running real imagenet test, logs are redirected to $TFB_DIR/fp16/imagenet.log"
  #real
  if [ -d "/tfrecords" ] ; then #if real data is not mounted skip the test
    ( time python tf_cnn_benchmarks.py \
                  --all_reduce_spec=nccl \
                  --data_format=NCHW \
                  --batch_size=$bsfp16 \
                  --model=resnet50 \
                  --optimizer=momentum \
                  --variable_update=replicated \
                  --nodistortions \
                  --gradient_repacking=8 \
                  --num_gpus=8 \
                  --num_batches=1000 \
                  --weight_decay=1e-4 \
                  --data_dir=/tfrecords/ \
                  --data_name=imagenet \
                  --use_fp16 ) &> $TFB_DIR/fp16/imagenet.log 2>&1
    return_code=$?
    #if there was an error, probably due to OOM, try reducing batch size
    if [ $return_code != 0 ]  ; then
      bsfp16=$(($bsfp16 / 2))
      echo "Reduce batch size for real data: $bsfp16"
      ( time python tf_cnn_benchmarks.py \
                    --all_reduce_spec=nccl \
                    --data_format=NCHW \
                    --batch_size=$bsfp16 \
                    --model=resnet50 \
                    --optimizer=momentum \
                    --variable_update=replicated \
                    --nodistortions \
                    --gradient_repacking=8 \
                    --num_gpus=8 \
                    --num_batches=1000 \
                    --weight_decay=1e-4 \
                    --data_dir=/tfrecords/ \
                    --data_name=imagenet \
                    --use_fp16 ) &> $TFB_DIR/fp16/imagenet.log
    fi
  else
    echo "ERROR: /tfrecords folder is not found" > $TFB_DIR/fp16/imagenet.log
  fi
  #synthetic 
  echo "Running synthetic data test, logs are redirected to $TFB_DIR/fp16/synthetic.log"
  ( time python tf_cnn_benchmarks.py \
                --all_reduce_spec=nccl \
                --num_gpus=8 \
                --batch_size=$bsfp16 \
                --num_batches=1000 \
                --model=resnet50 \
                --variable_update=replicated \
                --use_fp16 ) &> $TFB_DIR/fp16/synthetic.log
}

full_imagenet()
{
  #full imagenet training to 90 epoch with maximum batch size
  echo "Running full imagenet training, $TFB_DIR/full_imagenet/train_ep90_bs$bsfp16.log"
  #scale lr according to batch size: batch_size=256, lr=0.1
  lr1=$(echo "scale=6; $bsfp16 / 256 * 8 / 10" | bc )
  lr2=$(echo "scale=6; $lr1/10" | bc )
  lr3=$(echo "scale=6; $lr2/10" | bc )
  lr4=$(echo "scale=6; $lr3/10" | bc ) 
  #train
  ( time python tf_cnn_benchmarks.py \
                --all_reduce_spec=nccl \
                --data_format=NCHW \
                --batch_size=$bsfp16 \
                --model=resnet50 \
                --optimizer=momentum \
                --variable_update=replicated \
                --gradient_repacking=8 \
                --num_gpus=8 \
                --num_epochs=1 \
                --weight_decay=4e-5 \
                --data_dir=/tfrecords/ \
                --data_name=imagenet \
                --use_fp16 \
                --train_dir=/workspace/resnet50_train_full \
                --num_learning_rate_warmup_epochs=5 \
                --piecewise_learning_rate_schedule="$lr1;30;$lr2;60;$lr3;80;$lr4" ) &> $TFB_DIR/full_imagenet/train_ep90_bs$bsfp16.log 
  #eval
  python tf_cnn_benchmarks.py \
         --all_reduce_spec=nccl \
         --data_format=NCHW \
         --batch_size=$bsfp16 \
         --model=resnet50 \
         --optimizer=momentum \
         --variable_update=replicated \
         --gradient_repacking=8 \
         --num_gpus=8 \
         --num_epochs=1 \
         --weight_decay=4e-5 \
         --data_dir=/tfrecords/ \
         --data_name=imagenet \
         --use_fp16 \
         --train_dir=/workspace/resnet50_train_full \
         --num_learning_rate_warmup_epochs=5 \
         --piecewise_learning_rate_schedule="$lr1;30;$lr2;60;$lr3;80;$lr4" \
         --eval &> $TFB_DIR/full_imagenet/val_ep90_bs$bsfp16.log
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
  DATADIR=$WORKDIR/datasets/coco LOGDIR=/workspace/logs/object_detection  ./run.sub  
}

image_classification()
{
  cd /run_benchmarks/results/v0.5.0/nvidia/submission/code/image_classification/mxnet/
  docker pull 140.96.29.39:5000/myelintek/mlperf-nvidia:image_classification
  docker tag 140.96.29.39:5000/myelintek/mlperf-nvidia:image_classification mlperf-nvidia:image_classification
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
gpus_scalability_test
real_vs_synthetic_data
sleep 60s
full_imagenet
sleep 60s
rnn_translator
sleep 60s
object_detection
image_classification
sleep 10s
hwinfo
get_cuda_p2p
