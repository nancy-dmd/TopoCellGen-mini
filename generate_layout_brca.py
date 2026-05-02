import argparse
import json
import os
from collections import defaultdict
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import stopit
import torch as th
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def set_nonzero_to_one(array):
    return np.where(array != 0, 1, 0)


def save_cell_counts(save_path, generated_counts, ground_truth_counts):
    data = defaultdict(dict)
    for filename, counts in generated_counts.items():
        data[filename]["generated"] = counts
    for filename, counts in ground_truth_counts.items():
        data[filename]["ground_truth"] = counts

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


def build_parser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=True,
        sample_dir="",
        model_path="checkpoints/brca_m2c.pt",
        test_patch_path="data/brca_minimal/test_dataset",
        results_root_path="results",
        max_test_files=0,
        denoise_threshold=0.4,
        denoise_timeout=30,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def apply_brca_defaults(args):
    args.num_channels = 256
    args.num_res_blocks = 2
    args.num_head_channels = 64
    args.attention_resolutions = "32,16,8"
    args.class_cond = True
    args.use_scale_shift_norm = True
    args.resblock_updown = True
    args.use_fp16 = False
    args.learn_sigma = True
    args.diffusion_steps = 1000
    args.noise_schedule = "cosine"
    args.image_size = 256
    args.timestep_respacing = "ddim100"
    return args


def count_dots(cell_map):
    _, num_cells = ndi.label(cell_map)
    return num_cells


def count_cells_3c(cell_map):
    return [count_dots(cell_map[:, :, idx]) for idx in range(3)]


def visualize_cell_dot_map(dot_map, file_name):
    assert dot_map.shape[2] == 3, "Input must have 3 channels"
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    height, width, _ = dot_map.shape
    combined_image = np.zeros((height, width, 3))

    for idx, color in enumerate(colors):
        channel = dot_map[:, :, idx]
        for rgb_idx in range(3):
            combined_image[:, :, rgb_idx] += channel * color[rgb_idx]

    combined_image = np.clip(combined_image, 0, 1)
    plt.figure()
    plt.imshow(combined_image)
    plt.imsave(file_name, combined_image)
    plt.close()


def denoise_fun(inp, threshold):
    all_labels = np.zeros([256, 256, 3])
    unique_list = []
    for idx in range(3):
        image = (inp[0][idx] > threshold).numpy().astype(int)
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        if len(coords) > 0:
            mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image)
        for value in np.unique(labels):
            if np.sum(labels == value) <= 5:
                labels[labels == value] = 0
        all_labels[:, :, idx] = labels
        unique_list.append(len(np.unique(labels)))

    return all_labels, int(np.sum(unique_list)), unique_list


def build_results_dir(root_path):
    day = date.today().isoformat()
    current_time = datetime.now().strftime("%H-%M-%S")
    results_save_path = os.path.join(root_path, day, current_time)
    os.makedirs(results_save_path, exist_ok=True)
    return results_save_path


def main():
    args = apply_brca_defaults(build_parser().parse_args())

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    results_save_path = build_results_dir(args.results_root_path)
    with open(os.path.join(results_save_path, "hyperparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    test_patch_list = sorted(
        item for item in os.listdir(args.test_patch_path) if item.endswith(".npy")
    )
    if args.max_test_files and args.max_test_files > 0:
        test_patch_list = test_patch_list[: args.max_test_files]

    logger.log(f"using {len(test_patch_list)} test patches from {args.test_patch_path}")

    generated_cell_counts = {}
    ground_truth_cell_counts = {}

    save_npy_root_path = os.path.join(results_save_path, "npy")
    save_img_root_path = os.path.join(results_save_path, "img")
    os.makedirs(save_npy_root_path, exist_ok=True)
    os.makedirs(save_img_root_path, exist_ok=True)

    for test_patch_item in test_patch_list:
        test_patch_item_path = os.path.join(args.test_patch_path, test_patch_item)
        test_patch = np.load(test_patch_item_path)
        cell_count_list = count_cells_3c(test_patch)
        ground_truth_cell_counts[test_patch_item] = cell_count_list

        logger.log(
            f"generating {test_patch_item} with counts "
            f"{cell_count_list[0]}/{cell_count_list[1]}/{cell_count_list[2]}"
        )

        model_kwargs = {
            "y": th.tensor([cell_count_list], device=dist_util.dev(), dtype=th.float32)
        }
        sample = sample_fn(
            model,
            (1, 3, 256, 256),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        with stopit.ThreadingTimeout(args.denoise_timeout) as context_manager:
            labels_res, num_cell, cell_num_list = denoise_fun(
                sample.detach().cpu(), args.denoise_threshold
            )

        if context_manager.state == context_manager.EXECUTED:
            generated_cell_counts[test_patch_item] = cell_num_list
            test_patch_item_name = test_patch_item.split(".npy")[0]
            labels_res = set_nonzero_to_one(labels_res)

            np.save(
                os.path.join(save_npy_root_path, f"{test_patch_item_name}_gen_{num_cell}.npy"),
                labels_res,
            )
            img_path = os.path.join(save_img_root_path, f"{test_patch_item_name}_gen_{num_cell}.png")
            visualize_cell_dot_map(labels_res, img_path)
        else:
            logger.log(f"denoise timed out for {test_patch_item}")

    cell_counts_path = os.path.join(results_save_path, "cell_counts.json")
    save_cell_counts(cell_counts_path, generated_cell_counts, ground_truth_cell_counts)
    logger.log(f"generation done, outputs saved to {results_save_path}")


if __name__ == "__main__":
    main()
