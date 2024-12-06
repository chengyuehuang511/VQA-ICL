import argparse
import importlib
import json
import os
import uuid
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import utils
import math

from eval_datasets import VQADataset
from rices import RICES
from mmices import MMICES
from jices import JICES
from tqdm import tqdm

from eval_model import BaseEvalModel

from ok_vqa_utils import postprocess_ok_vqa_generation
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation

from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    help="Model name. Currently only `OpenFlamingo` is supported.",
    default="open_flamingo",
)
parser.add_argument(
    "--model_id",
    type=str,
    help="Model ID for the model to use for JICES.",
    default="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
)
parser.add_argument(
    "--results_file", type=str, default=None, help="JSON file to save results"
)

# Trial arguments
parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of trials to run for each shot using different demonstrations",
)
parser.add_argument(
    "--trial_seeds",
    nargs="+",
    type=int,
    default=[42],
    help="Seeds to use for each trial for picking demonstrations and eval sets",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=-1,
    help="Number of samples to evaluate on. -1 for all samples.",
)
parser.add_argument(
    "--query_set_size", type=int, default=2048, help="Size of demonstration query set"
)

parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument(
    "--no_caching_for_classification",
    action="store_true",
    help="Whether to skip using key-value caching for classification evals, which usually speeds it up.",
)
parser.add_argument(
    "--classification_prompt_ensembling",
    action="store_true",
    help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
)
parser.add_argument(
    "--embedding_selection",
    type=str,
    default=None,
    choices=["rices", "mmices", "jices"],
    help="Which embedding selection method to use.",
)
parser.add_argument(
    "--rices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--rices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for RICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--mmices_vision_encoder_path",
    default="ViT-L-14",
    type=str,
    help="CLIP vision encoder to use for MMICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--mmices_vision_encoder_pretrained",
    default="openai",
    type=str,
    help="CLIP vision encoder to use for MMICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--mmices_lm_path",
    default="anas-awadalla/mpt-1b-redpajama-200b",
    type=str,
    help="Language model to use for MMICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--mmices_lm_tokenizer_path",
    default="anas-awadalla/mpt-1b-redpajama-200b",
    type=str,
    help="Language model to use for MMICES if cached_demonstration_features is None.",
)
parser.add_argument(
    "--cached_demonstration_features",
    default="/coc/testnvme/chuang475/projects/VQA-ICL/cache",
    help="Directory where rices features for all choices of in-context examples are stored as a pkl file with the dataset name. If None, features are re-computed by script.",
)

# Per-dataset evaluation flags
# parser.add_argument(
#     "--eval_vqav2",
#     action="store_true",
#     default=False,
#     help="Whether to evaluate on VQAV2.",
# )
# parser.add_argument(
#     "--eval_ok_vqa",
#     action="store_true",
#     default=False,
#     help="Whether to evaluate on OK-VQA.",
# )
# parser.add_argument(
#     "--eval_vizwiz",
#     action="store_true",
#     default=False,
#     help="Whether to evaluate on VizWiz.",
# )
# parser.add_argument(
#     "--eval_textvqa",
#     action="store_true",
#     default=False,
#     help="Whether to evaluate on TextVQA.",
# )

parser.add_argument(
    "--train_dataset_name",
    type=str,
    default=None,
    help="Name of the dataset to use for training the model.",
)

parser.add_argument(
    "--test_dataset_name",
    type=str,
    default=None,
    help="Name of the dataset to use for evaluating the model.",
)

# Dataset arguments

## VQAV2 Dataset
parser.add_argument(
    "--vqav2_train_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_train_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_image_dir_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_questions_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_test_annotations_json_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--vqav2_final_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_test2015_questions.json file containing all test questions. This is required to format the predictions for EvalAI.",
    default=None,
)

## OK-VQA Dataset
parser.add_argument(
    "--ok_vqa_train_image_dir_path",
    type=str,
    help="Path to the vqav2/train2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_train2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_train_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_train2014_annotations.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_image_dir_path",
    type=str,
    help="Path to the vqav2/val2014 directory.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_questions_json_path",
    type=str,
    help="Path to the v2_OpenEnded_mscoco_val2014_questions.json file.",
    default=None,
)
parser.add_argument(
    "--ok_vqa_test_annotations_json_path",
    type=str,
    help="Path to the v2_mscoco_val2014_annotations.json file.",
    default=None,
)

## VizWiz Dataset
parser.add_argument(
    "--vizwiz_train_image_dir_path",
    type=str,
    help="Path to the vizwiz train images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_image_dir_path",
    type=str,
    help="Path to the vizwiz test images directory.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_train_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_questions_json_path",
    type=str,
    help="Path to the vizwiz questions json file.",
    default=None,
)
parser.add_argument(
    "--vizwiz_test_annotations_json_path",
    type=str,
    help="Path to the vizwiz annotations json file.",
    default=None,
)

# TextVQA Dataset
parser.add_argument(
    "--textvqa_image_dir_path",
    type=str,
    help="Path to the textvqa images directory.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_train_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_questions_json_path",
    type=str,
    help="Path to the textvqa questions json file.",
    default=None,
)
parser.add_argument(
    "--textvqa_test_annotations_json_path",
    type=str,
    help="Path to the textvqa annotations json file.",
    default=None,
)

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)


def main():
    args, leftovers = parser.parse_known_args()
    module = importlib.import_module(f"models.{args.model}")

    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    eval_model = module.EvalModel(model_args)

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    eval_model.set_device(device_id)
    eval_model.init_distributed()  # DDP
    print("Device ID: ", eval_model.device)

    if args.model != "open_flamingo" and args.shots != [0]:
        raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    print(f"Support Set: {args.train_dataset_name}, Query Set: {args.test_dataset_name}")

    for shot in args.shots:
        scores = []
        for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
            score = evaluate_vqa(
                args=args,
                eval_model=eval_model,
                num_shots=shot,
                seed=seed,
                dataset_name=args.test_dataset_name,
                train_dataset_name=args.train_dataset_name,
            )
            if args.rank == 0:
                print(f"Shots {shot} Trial {trial} {args.test_dataset_name} score: {score}")
                scores.append(score)

        if args.rank == 0:
            print(f"Shots {shot} Mean {args.test_dataset_name} score: {np.nanmean(scores)}")
            results[args.test_dataset_name].append(
                {
                    "shots": shot,
                    "trials": scores,
                    "mean": np.nanmean(scores),
                    "stddev": np.nanstd(scores),
                }
            )

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
    train_dataset_name=None,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        # train_image_dir_path = args.ok_vqa_train_image_dir_path
        # train_questions_json_path = args.ok_vqa_train_questions_json_path
        # train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        # train_image_dir_path = args.vqav2_train_image_dir_path
        # train_questions_json_path = args.vqav2_train_questions_json_path
        # train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif dataset_name == "vizwiz":
        # train_image_dir_path = args.vizwiz_train_image_dir_path
        # train_questions_json_path = args.vizwiz_train_questions_json_path
        # train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        # train_image_dir_path = args.textvqa_image_dir_path
        # train_questions_json_path = args.textvqa_train_questions_json_path
        # train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if train_dataset_name is None:
        train_dataset_name = dataset_name

    cached_features_path = f"{args.cached_demonstration_features}/{args.embedding_selection}/train/{train_dataset_name}.pkl"
    query_cached_features_path = f"{args.cached_demonstration_features}/{args.embedding_selection}/test/{dataset_name}.pkl"

    if train_dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        # test_image_dir_path = args.ok_vqa_test_image_dir_path
        # test_questions_json_path = args.ok_vqa_test_questions_json_path
        # test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif train_dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        # test_image_dir_path = args.vqav2_test_image_dir_path
        # test_questions_json_path = args.vqav2_test_questions_json_path
        # test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif train_dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        # test_image_dir_path = args.vizwiz_test_image_dir_path
        # test_questions_json_path = args.vizwiz_test_questions_json_path
        # test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif train_dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        # test_image_dir_path = args.textvqa_image_dir_path
        # test_questions_json_path = args.textvqa_test_questions_json_path
        # test_annotations_json_path = args.textvqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {train_dataset_name}")

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=train_dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = utils.compute_effective_num_shots(num_shots, args.model)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
    )
    # for batch in test_dataloader:
    #     print("rank in main", args.rank)
    #     print("len(batch['image'])", len(batch["image"]))

    if args.embedding_selection == "rices":
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features_path=cached_features_path,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
            query_dataset=test_dataset,
            query_cached_features_path=query_cached_features_path,
        )
    elif args.embedding_selection == "mmices":
        rices_dataset = MMICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features_path=cached_features_path,
            vision_encoder_path=args.mmices_vision_encoder_path,
            vision_encoder_pretrained=args.mmices_vision_encoder_pretrained,
            lm_path=args.mmices_lm_path,
            lm_tokenizer_path=args.mmices_lm_tokenizer_path,
        )
    elif args.embedding_selection == "jices":
        rices_dataset = JICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features_path=cached_features_path,
            eval_model=eval_model,
        )
    else:
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    utils.random_seed(seed, args.rank)
    predictions = []
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        if args.embedding_selection == "rices":
            batch_demo_samples = rices_dataset.find(batch, effective_num_shots)
        elif args.embedding_selection == "mmices":
            batch_demo_samples = rices_dataset.find(batch, effective_num_shots, K=200)
        elif args.embedding_selection == "jices":
            batch_demo_samples = rices_dataset.find(batch, effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    + "\n"
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        process_function = (
            postprocess_ok_vqa_generation
            if dataset_name == "ok_vqa"
            else postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id})

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists

    if args.rank != 0:
        return None

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            f"{dataset_name}results_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")

    return acc


if __name__ == "__main__":
    main()