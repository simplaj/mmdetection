forked form mmdet
cv hw, ai collegue of NKU
use crowddet train widerperson based on mmdet
main change:
  configs/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman.py
  mmdet/datasets/crowdhuman.py
  mmdet/evaluation/metrics/crowdhuman_metric.py
  tools/dataset_converters/wider_person2ch.py
  tools/visual.py
