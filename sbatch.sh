#!/bin/bash
cd /coc/testnvme/chuang475/projects/VQA-ICL
name="run_eval"
embedding_selection="rices"

for dataset_name in "ok_vqa" #"vqav2" "ok_vqa" "textvqa" "vizwiz"
do
    job_name="${name}_$(date +%Y%m%d_%H%M%S)"
    output_dir="output/${job_name}"
    mkdir -p "$output_dir"
    sbatch --export "ALL,embedding_selection=${embedding_selection},dataset_name=${dataset_name}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" ${name}.sh
done