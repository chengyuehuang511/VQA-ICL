#!/bin/bash                   
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt
#SBATCH --mem-per-gpu=45G

<<com
Example Slurm evaluation script. 
Notes:
- VQAv2 test-dev and test-std annotations are not publicly available. 
  To evaluate on these splits, please follow the VQAv2 instructions and submit to EvalAI.
  This script will evaluate on the val split.
com

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

cd /coc/testnvme/chuang475/projects/VQA-ICL

export PYTHONPATH="$PYTHONPATH:open_flamingo"
srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=8 evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path "/nethome/chuang475/flash/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt" \
    --results_file "results.json" \
    --precision amp_bf16 \
    --batch_size 16 \
    --eval_ok_vqa \
    --vqav2_train_image_dir_path "/srv/datasets/coco/train2014" \
    --vqav2_train_annotations_json_path "/srv/datasets/vqa2.0/v2_mscoco_train2014_annotations.json" \
    --vqav2_train_questions_json_path "/srv/datasets/vqa2.0/v2_OpenEnded_mscoco_train2014_questions.json" \
    --vqav2_test_image_dir_path "/srv/datasets/coco/val2014" \
    --vqav2_test_annotations_json_path "/srv/datasets/vqa2.0/v2_mscoco_val2014_annotations.json" \
    --vqav2_test_questions_json_path "/srv/datasets/vqa2.0/v2_OpenEnded_mscoco_val2014_questions.json" \
    --ok_vqa_train_image_dir_path "/srv/datasets/coco/train2014" \
    --ok_vqa_train_annotations_json_path "/srv/datasets/ok-vqa_dataset/mscoco_train2014_annotations.json" \
    --ok_vqa_train_questions_json_path "/srv/datasets/ok-vqa_dataset/OpenEnded_mscoco_train2014_questions.json" \
    --ok_vqa_test_image_dir_path "/srv/datasets/coco/val2014" \
    --ok_vqa_test_annotations_json_path "/srv/datasets/ok-vqa_dataset/mscoco_val2014_annotations.json" \
    --ok_vqa_test_questions_json_path "/srv/datasets/ok-vqa_dataset/OpenEnded_mscoco_val2014_questions.json" \
    --textvqa_image_dir_path "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/train_images/" \
    --textvqa_train_questions_json_path "data/textvqa/train_questions_vqa_format.json" \
    --textvqa_train_annotations_json_path "data/textvqa/train_annotations_vqa_format.json" \
    --textvqa_test_questions_json_path "data/textvqa/val_questions_vqa_format.json" \
    --textvqa_test_annotations_json_path "data/textvqa/val_annotations_vqa_format.json" \
    --vizwiz_train_image_dir_path "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/images/train" \
    --vizwiz_test_image_dir_path "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/images/val" \
    --vizwiz_train_questions_json_path "data/vizwiz/train_questions_vqa_format.json" \
    --vizwiz_train_annotations_json_path "data/vizwiz/train_annotations_vqa_format.json" \
    --vizwiz_test_questions_json_path "data/vizwiz/val_questions_vqa_format.json" \
    --vizwiz_test_annotations_json_path "data/vizwiz/val_annotations_vqa_format.json" \
