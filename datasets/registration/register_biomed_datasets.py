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
datasets = [
    'Task09_Spleen', 'Task08_HepaticVessel', 'Task07_Pancreas', 'Task06_Lung',
    'Task05_Prostate', 'Task04_Hippocampus', 'Task03_Liver', 'Task02_Heart',
    'Task01_BrainTumour', 'UWaterlooSkinCancer', 'kits23', 'Radiography_Viral_Pneumonia',
    'Radiography_Normal', 'Radiography_Lung_Opacity', 'Radiography_COVID',
    'siim-acr-pneumothorax', 'REFUGE', 'QaTa-COV19', 'PolypGen', 'PanNuke',
    'OCT-CME', 'NeoPolyp', 'MMs', 'LiverUS', 'LIDC-IDRI', 'LGG', 'GlaS', 'G1020',
    'FH-PS-AOP', 'DRIVE', 'CXR_Masks_and_Labels', 'ISIC', 'COVID-QU-Ex',
    'COVID-19_CT', 'CDD-CESM', 'amos_22_MRI', 'amos_22_CT', 'CAMUS', 'BreastUS',
    'ACDC'
]   # provide name of the dataset under biomedparse_datasets
splits = ['train', 'test']    # provide split name, e.g., train, test, val. Update this if your splits are different.

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
        list[str]: a list of class names if found in the json, otherwise None.
    """

    with PathManager.open(annot_json) as f:
        json_info = json.load(f)
        
    # <<< --- Extract Category Names --- >>>
    category_names = None
    if "categories" in json_info:
        try:
            # Sort by id to ensure order matches class indices
            sorted_categories = sorted(json_info["categories"], key=lambda x: x["id"])
            # Extract the name field
            category_names = [cat["name"] for cat in sorted_categories]
            logger.info(f"Found {len(category_names)} categories in {annot_json}")
        except KeyError as e:
            logger.warning(f"'categories' key found in {annot_json}, but couldn't extract 'id' or 'name'. {e}")
        except Exception as e:
            logger.warning(f"Error processing 'categories' in {annot_json}: {e}")
    # <<< --- End Extraction --- >>>
        
    # build dictionary for grounding
    grd_dict = collections.defaultdict(list)
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)

    mask_root = image_root + '_mask'
    ret = []
    for image in json_info["images"]:
        image_id = int(image["id"])
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]
        for ann in grounding_anno:
            if 'mask_file' not in ann:
                ann['mask_file'] = image['file_name']
            ann['mask_file'] = os.path.join(mask_root, ann['mask_file'])
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "grounding_info": [ann],
                    "sem_seg_file_name": ann['mask_file'],
                }
            )
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]

    return ret, category_names


def register_biomed(
    name, metadata, image_root, annot_json):
    # <<< --- Load data and potentially class names --- >>>
    dataset_dicts, class_names = load_biomed_json(image_root, annot_json, metadata)

    # <<< --- Register dataset function --- >>>
    DatasetCatalog.register(
        name,
         lambda: dataset_dicts, # Pass the loaded data directly
    )

    # <<< --- Set metadata, including class names if found --- >>>
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        # Add stuff_classes if they were extracted
        **(dict(stuff_classes=class_names) if class_names else {}),
        **metadata,
    )


def register_all_biomed(root):
    for (
        prefix,
        (image_root, annot_root),
    ) in _PREDEFINED_SPLITS_BIOMED.items():
        # <<< --- Load metadata once --- >>>
        metadata = get_metadata()
        register_biomed(
            prefix,
            metadata, # Pass the loaded metadata
            os.path.join(root, image_root),
            os.path.join(root, annot_root),
        )


# <<< --- Add logger definition --- >>>
logger = logging.getLogger(__name__)
# <<< --- End Add logger --- >>>

_root = os.getenv("DATASET", "datasets")
register_all_biomed(_root)
