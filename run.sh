cd /workspace
mkdir -p mlperf
mkdir -p logs
cd /run_benchmarks/benchmarks/scripts/tf_cnn_benchmarks
script -c "time python tf_cnn_benchmarks.py --num_gpus=4 --batch_size=32 --use_fp16 --model=resnet50 --variable_update=parameter_server" /workspace/logs/benchmark_resnet50_synthetic.log
if [ -d "/tfrecords" ] ; then
  script -c "time python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=32 --model=resnet50 --optimizer=momentum --variable_update=replicated --nodistortions --gradient_repacking=8 --num_gpus=4 --num_epochs=90 --weight_decay=1e-4 --data_dir=/tfrecords/ --data_name=imagenet --use_fp16 --train_dir=/workspace/resnet50_train" /workspace/logs/benchmark_resnet50_imagenet.log
fi

cd /workspace/mlperf
if [ ! -d "/workspace/mlperf/training" ] ; then
  git clone https://github.com/mlperf/training.git
fi
#object detection with coco
cd training/object_detection/
cp /run_and_time_obj_detect.sh .
docker pull 140.96.29.39:5000/mlperf/object_detection:latest
if [ ! -d "pytorch/datasets/coco" ] ; then
  source download_dataset.sh
fi
script -c 'docker run --runtime nvidia -v $WORKDIR:/workspace -e MASTER_ADDR=localhost -e MASTER_PORT=4000 -it --rm --ipc=host 140.96.29.39:5000/mlperf/object_detection:latest bash -c "cd mlperf/training/object_detection && ./install.sh && bash run_and_time_obj_detect.sh"' /workspace/logs/mlperf_object_detection.log
