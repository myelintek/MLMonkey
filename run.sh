#!/bin/bash
cd /workspace
mkdir -p mlperf
mkdir -p logs
cd logs
mkdir -p gpu_scalability fp16 COCO
cd /run_benchmarks/benchmarks/scripts/tf_cnn_benchmarks
#find max batch size for fp16 and fp32
bs1gpu=( 1024 512 256 128 64 32 16 )
for bsize in ${bs1gpu[@]}
do
  python tf_cnn_benchmarks.py --num_gpus=6 --batch_size=$bsize --num_batches=10 --use_fp16 --model=resnet50 --variable_update=replicated
  test $? -eq 0 && bsfp16=$bsize && break
done

for bsize in ${bs1gpu[@]}
do
  python tf_cnn_benchmarks.py --num_gpus=6 --batch_size=$bsize --num_batches=10 --model=resnet50 --variable_update=replicated
  test $? -eq 0 && bsfp32=$bsize && break
done

#gpus scalability test
num_gpus=( 1 2 4 6 ) #TODO change to ( 1 2 4 8 )

for gpus in ${num_gpus[@]}
do
  script -c "time -f '\t%e real,\t%U user,\t%S sys' python tf_cnn_benchmarks.py --num_gpus=$gpus --batch_size=$bsfp32 --num_batches=10 --model=resnet50 --variable_update=replicated --all_reduce_spec=nccl" /workspace/logs/gpu_scalability/$gpus.log
done

#fp16 test #TODO change --num_gpus=8
script -c "time -f '\t%e real,\t%U user,\t%S sys' python tf_cnn_benchmarks.py --num_gpus=6 --batch_size=$bsfp16 --num_batches=10 --model=resnet50 --variable_update=replicated" /workspace/logs/fp16/synthetic.log
if [ -d "/tfrecords" ] ; then
  script -c "time -f '\t%e real,\t%U user,\t%S sys' python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=$bsfp16 --model=resnet50 --optimizer=momentum --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=6 --num_epochs=90 --weight_decay=1e-4 --data_dir=/tfrecords/ --data_name=imagenet --use_fp16 --train_dir=/workspace/resnet50_train" /workspace/logs/fp16/imagenet.log
fi

#mlperf COCO	
cd /workspace/mlperf
if [ ! -d "/workspace/mlperf/training" ] ; then
  git clone https://github.com/mlperf/training.git
fi
cd training/object_detection/
cp /run_and_time_obj_detect.sh .
chmod +x run_and_time_obj_detect.sh
docker pull 140.96.29.39:5000/mlperf/object_detection:latest
if [ ! -d "pytorch/datasets/coco" ] ; then
  source download_dataset.sh
fi
script -c "docker run --runtime nvidia -v $WORKDIR:/workspace -e MASTER_ADDR=localhost -e MASTER_PORT=4000 -e TIMEFORMAT='%E real,  %U user,  %S sys' -it --rm --ipc=host 140.96.29.39:5000/mlperf/object_detection:latest bash -c 'cd mlperf/training/object_detection && ./install.sh && ./run_and_time_obj_detect.sh'" /workspace/logs/COCO/mlperf_object_detection.log
