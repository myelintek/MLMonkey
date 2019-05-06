export WORLD_SIZE=8
pushd pytorch
for ((rank=0;rank<WORLD_SIZE;rank++)); do
  time  RANK=$rank python tools/train_mlperf.py --local_rank $rank --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025 &
done
popd
