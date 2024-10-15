python cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 128 \
  --eval_coco \
  --coco_train_image_dir_path /path/to/coco/train2014 \
  --coco_val_image_dir_path /path/to/coco/val2014 \
  --coco_karpathy_json_path /path/to/coco/dataset_coco.json \
  --coco_annotations_json_path /path/to/coco/annotations/captions_train2014.json \
  --output_dir /path/to/coco/features 

