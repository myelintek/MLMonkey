cd /workspace
mkdir -p mlperf
mkdir -p logs
cd mlperf
if [ ! -d "/workspace/mlperf/training" ]; then
  git clone https://github.com/mlperf/training.git
fi
#object detection with coco
cd training/object_detection/
docker pull 140.96.29.39:5000/mlperf/object_detection:latest
if [ ! -d "pytorch/datasets/coco" ]; then
  source download_dataset.sh
fi
script -c 'docker run --runtime nvidia -v $WORKDIR:/workspace -it --rm --ipc=host mlperf/object_detection bash -c "cd mlperf/training/object_detection && ./install.sh && ./run_and_time.sh"' /workspace/logs/mlperf_object_detection.log
