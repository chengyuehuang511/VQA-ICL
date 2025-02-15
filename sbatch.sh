#!/bin/bash
cd /coc/testnvme/chuang475/projects/VQA-ICL
name="run_eval"

for embedding_selection in "jices" #"None" #"rices" "mmices" "jices"
do
    for train_dataset_name in "ok_vqa" #"vizwiz" #"vqav2" "ok_vqa" "textvqa" "vizwiz"
    do
        for test_dataset_name in "ok_vqa" #"vizwiz" #"vqav2" "ok_vqa" "textvqa" "vizwiz"
        do
            # if [ "$train_dataset_name" == "$test_dataset_name" && "$train_dataset_name" == "ok_vqa" ]; then
            #     continue
            # fi
            job_name="${name}_$(date +%Y%m%d_%H%M%S)"
            output_dir="output/idefics2/pai/train_${train_dataset_name}/test_${test_dataset_name}/${embedding_selection}/${job_name}"
            mkdir -p "$output_dir"
            sbatch --export "ALL,embedding_selection=${embedding_selection},test_dataset_name=${test_dataset_name},train_dataset_name=${train_dataset_name}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" ${name}.sh
        done
    done
done