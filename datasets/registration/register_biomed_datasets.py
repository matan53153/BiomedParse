# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import json
import os
import collections
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS_BIOMED = {}

# example of registering a dataset
datasets = ['UWaterlooSkinCancer']   # provide name of the dataset under biomedparse_datasets
splits = ['train','test']    # provide split name, e.g., train, test, val. Here there is only one 'demo' split in the example demo dataset

# Here we register all the splits of the dataset
for name in datasets:
    for split in splits:
        dataname = f'biomed_{name.replace("/", "-")}_{split}'
        image_root = f"{name}/{split}"
        ann_root = f"{name}/{split}.json"
        _PREDEFINED_SPLITS_BIOMED[dataname] = (image_root, ann_root)
# The resulting dataset name is: biomed_BiomedParseData-Demo_demo

# # Add your dataset here
# datasets = ['YOUR_DATASET_NAME', ]   # provide name of the dataset under biomedparse_datasets
# splits = ['train', 'test']    # provide split name, e.g., train, test, val

# # Here we register all the splits of the dataset
# for name in datasets:
#     for split in splits:
#         dataname = f'biomed_{name.replace("/", "-")}_{split}'
#         image_root = f"{name}/{split}"
#         ann_root = f"{name}/{split}.json"
#         _PREDEFINED_SPLITS_BIOMED[dataname] = (image_root, ann_root)
# # The resulting dataset names are: biomed_YOUR_DATASET_NAME_train, biomed_YOUR_DATASET_NAME_test


def get_metadata():
    meta = {}
    return meta


def load_biomed_json(image_root, annot_json, metadata):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    import logging

    with PathManager.open(annot_json) as f:
        json_info = json.load(f)

    # build dictionary for grounding - used later
    grd_dict = collections.defaultdict(list)
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)

    mask_root = image_root + '_mask'
    ret = []
    image_id_to_info = {}

    # First pass: Populate image_id_to_info with basic image data
    for image in json_info["images"]:
        image_id = int(image["id"])
        image_id_to_info[image_id] = {
            "file_name": os.path.join(image_root, image['file_name']),
            "image_id": image_id,
            "sem_seg_file_name": None, # Initialize as None, will be filled from annotation
            "grounding_info": []
        }

    # Second pass: Process annotations to get grounding info AND infer sem_seg_file_name
    processed_image_ids_for_sem_seg = set() # Keep track of images for which we've set sem_seg path

    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        if image_id in image_id_to_info:
            # --- WORKAROUND START ---
            # If we haven't assigned a sem_seg_file_name for this image yet,
            # use the mask_file from this (first encountered) annotation.
            if image_id not in processed_image_ids_for_sem_seg:
                if "mask_file" in grd_ann and grd_ann["mask_file"] is not None:
                    # Construct the full path using the mask filename from the annotation
                    # IMPORTANT: Assuming the mask_file in annotation is just the filename, not a path
                    sem_seg_path = os.path.join(mask_root, grd_ann["mask_file"])
                    image_id_to_info[image_id]["sem_seg_file_name"] = sem_seg_path
                    processed_image_ids_for_sem_seg.add(image_id)
                    # Add a log warning about this workaround (log only once per dataset load)
                    if len(processed_image_ids_for_sem_seg) == 1:
                        logging.warning(f"WORKAROUND: Inferring 'sem_seg_file_name' from first annotation's 'mask_file' for image_id {image_id} due to missing key in JSON 'images' list. Assumed path: {sem_seg_path}")
                else:
                    # Log error if the first annotation also lacks mask_file
                    logging.error(f"Cannot infer 'sem_seg_file_name' for image_id {image_id}: 'mask_file' missing or None in first annotation.")
            # --- WORKAROUND END ---

            # Process grounding mask path (ensure it's relative to mask_root)
            # Note: This might overwrite grd_ann['mask_file'] if it was already a full path, assumes it's filename only
            if 'mask_file' in grd_ann and grd_ann['mask_file'] is not None:
                 current_mask_filename = os.path.basename(grd_ann['mask_file']) # Extract filename just in case
                 grd_ann['mask_file'] = os.path.join(mask_root, current_mask_filename)
            else:
                 # Handle missing grounding mask? Set to None or log error.
                 grd_ann['mask_file'] = None # Example: set to None

            image_id_to_info[image_id]["grounding_info"].append(grd_ann)
        else:
            logging.warning(f"Annotation references image_id {image_id} which is not found in image list.")

    # Add check for images that had no annotations at all to infer from
    for image_id, info in image_id_to_info.items():
        if info["sem_seg_file_name"] is None:
             logging.error(f"Failed to find any valid annotation for image_id {image_id} to infer 'sem_seg_file_name'.")

    # Convert the dictionary values to a list
    ret = list(image_id_to_info.values())

    assert len(ret), f"No images found in {image_root}!"
    if ret:
        assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
        # Check if the sem_seg path exists *if* it was successfully read and not None
        if ret[0]["sem_seg_file_name"] is not None:
             assert PathManager.isfile(ret[0]["sem_seg_file_name"]), f"Semantic mask file not found: {ret[0]['sem_seg_file_name']}"
    return ret


def register_biomed(
    name, metadata, image_root, annot_json):
    DatasetCatalog.register(
        name,
        lambda: load_biomed_json(image_root, annot_json, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_biomed(root):
    for (
        prefix,
        (image_root, annot_root),
    ) in _PREDEFINED_SPLITS_BIOMED.items():
        register_biomed(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, annot_root),
        )


_root = os.getenv("DATASET", "datasets")
register_all_biomed(_root)
