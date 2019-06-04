#!/bin/bash
#record hardware spec
mkdir -p /web
lshw -html > /web/specs.html
mkdir -p /tmp/logs
cd /workspace
mkdir -p mlperf
mkdir -p logs
cd logs
mkdir -p gpu_scalability fp16 COCO
cd /run_benchmarks/benchmarks/scripts/tf_cnn_benchmarks
#find max batch size for fp16 and fp32
bs1gpu=( 1024 512 256 128 64 32 16 8 4 )
echo "Running Imagenet with Resnet50 benchmark"
echo "Checking maximum possible batch size for fp16 with synthetic data"
for bsize in ${bs1gpu[@]}
do
  python tf_cnn_benchmarks.py --num_gpus=8 --batch_size=$bsize --num_batches=10 --model=resnet50 --variable_update=replicated --use_fp16 --data_format=NCHW --optimizer=momentum --gradient_repacking=8  &> /tmp/logs/find_max_bs_fp16.log
  test $? -eq 0 && bsfp16=$bsize && echo "Maximum batch size for fp16 with synthetic data: $bsize" && break
#  echo "ERROR: maximum batch size for fp16 synthetic data not found"
done

echo "Checking maximum possible batch size for fp32 with synthetic data"
for bsize in ${bs1gpu[@]}
do
  python tf_cnn_benchmarks.py --num_gpus=8 --batch_size=$bsize --num_batches=10 --model=resnet50 --variable_update=replicated  &> /tmp/logs/find_max_bs_fp32.log
  test $? -eq 0 && bsfp32=$bsize &&  echo "Maximum batch size for fp32 with synthetic data: $bsize" && break
#  echo "ERROR: maximum batch size for fp32 synthetic data not found"
done

export TIMEFORMAT='%E real,  %U user,  %S sys'

#gpus scalability test
num_gpus=( 1 2 4 8 ) 
for gpus in ${num_gpus[@]}
do
  echo "Running gpu scalability test, output is redirected to /workspace/logs/gpu_scalability/$gpus.log"
  ( time python tf_cnn_benchmarks.py --num_gpus=$gpus --batch_size=$bsfp32 --num_epochs=1 --model=resnet50 --variable_update=replicated --all_reduce_spec=nccl ) &> /workspace/logs/gpu_scalability/$gpus.log
done

#fp16 real data
echo "Running real imagenet test, logs are redirected to /workspace/logs/fp16/imagenet.log"
if [ -d "/tfrecords" ] ; then
  ( time python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=$bsfp16 --model=resnet50 --optimizer=momentum --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=8 --num_batches=1000 --weight_decay=1e-4 --data_dir=/tfrecords/ --data_name=imagenet --use_fp16 --train_dir=/workspace/resnet50_train ) &> /workspace/logs/fp16/imagenet.log 2>&1
  return_code=$?
  echo $return_code
  echo $bsfp16
  #if there was an error, probably due to OOM, try reducing batch size
  if [ $return_code != 0 ]  ; then
    bsfp16=$(($bsfp16 / 2))
    echo "Reduce batch size for real data: $bsfp16"
    ( time python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=$bsfp16 --model=resnet50 --optimizer=momentum --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=8 --num_batches=1000 --weight_decay=1e-4 --data_dir=/tfrecords/ --data_name=imagenet --use_fp16 --train_dir=/workspace/resnet50_train ) &> /workspace/logs/fp16/imagenet.log
  fi
else
  echo "ERROR: /tfrecords folder is not found" > /workspace/logs/fp16/imagenet.log
fi

#fp16 synthetic 
echo "Running synthetic data test, logs are redirected to /workspace/logs/fp16/synthetic.log"
( time python tf_cnn_benchmarks.py --num_gpus=8 --batch_size=$bsfp16 --num_batches=1000 --model=resnet50 --variable_update=replicated --use_fp16 ) &> /workspace/logs/fp16/synthetic.log

#full imagenet training to 90 epoch with maximum batch size
mkdir -p /workspace/logs/full_imagenet
echo "Running full imagenet training, /workspace/logs/full_imagenet/train_ep90_bs$bsfp16.log"
lr= $(( $bsfp16 / 256 * 8 * 0.1 ))
( time python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=$bsfp16 --model=resnet50 --optimizer=momentum --variable_update=replicated --gradient_repacking=8 --num_gpus=8 --num_epochs=90 --weight_decay=4e-5 --data_dir=/tfrecords/ --data_name=imagenet --use_fp16 --train_dir=/workspace/resnet50_train_full --num_learning_rate_warmup_epochs=5 --piecewise_learning_rate_schedule="$lr;30;$(($lr * 0.1));60;$(($lr * 0.01));80;$(($lr * 0.001))" ) &> /workspace/logs/full_imagenet/train_ep90_bs$bsfp16.log 

python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=$bsfp16 --model=resnet50 --optimizer=momentum --variable_update=replicated --gradient_repacking=8 --num_gpus=8 --num_epochs=90 --weight_decay=4e-5 --data_dir=/tfrecords/ --data_name=imagenet --use_fp16 --train_dir=/workspace/resnet50_train_full --num_learning_rate_warmup_epochs=5 --piecewise_learning_rate_schedule="$lr;30;$(($lr * 0.1));60;$(($lr * 0.01));80;$(($lr * 0.001))" --eval &> /workspace/logs/full_imagenet/val_ep90_bs$bsfp16.log

#nvidia-optimized mlperf, default parameters
#rnn_translator
cd /run_benchmarks/results/v0.5.0/nvidia/submission/code/rnn_translator
bash download_dataset.sh



#mlperf COCO	
#cd /workspace/mlperf
#if [ ! -d "/workspace/mlperf/training" ] ; then
#  git clone https://github.com/mlperf/training.git
#fi
#cd training/object_detection/
#cp /scripts/run_and_time_obj_detect.sh .
#chmod +x run_and_time_obj_detect.sh
#docker pull 140.96.29.39:5000/mlperf/object_detection:latest
#if [ ! -d "pytorch/datasets/coco" ] ; then
#  echo "Downloading COCO dataset"
#  source download_dataset.sh
#fi
#echo "Running object detection training on COCO"
#docker run --runtime nvidia -v $WORKDIR:/workspace -e MASTER_ADDR=localhost -e MASTER_PORT=4000 -e TIMEFORMAT="$TIMEFORMAT" -it --rm --ipc=host 140.96.29.39:5000/mlperf/object_detection:latest bash -c 'cd mlperf/training/object_detection && ./install.sh && ./run_and_time_obj_detect.sh' &> /workspace/logs/COCO/mlperf.log

#parse logs to csv
mkdir /web/csv
python /scripts/parser.py
