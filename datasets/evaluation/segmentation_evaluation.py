# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator
from utilities.distributed import synchronize

from ..semseg_loader import load_semseg
from detectron2.data import detection_utils as utils

logger = logging.getLogger(__name__)


class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None

        # --- Handle missing stuff_classes ---
        if hasattr(meta, "stuff_classes"):
            self._class_names = meta.stuff_classes
        elif hasattr(meta, "thing_classes"):
            self._logger.warning(f"Metadata for '{dataset_name}' missing 'stuff_classes', using 'thing_classes' instead.")
            self._class_names = meta.thing_classes
        else:
            # Fallback: Generate generic names up to ignore_label if possible
            ignore_label = getattr(meta, "ignore_label", 255) # Get ignore_label safely
            # Infer num_classes, common convention is labels 0 to N-1, with N being ignore_label
            num_classes = ignore_label if isinstance(ignore_label, int) and ignore_label > 0 else 16 # Default to 16 if ignore_label invalid
            self._class_names = [f"class_{i}" for i in range(num_classes)]
            self._class_names.append("background") # Add background if ignore_label is used
            self._logger.warning(
                f"Metadata for '{self._dataset_name}' missing 'stuff_classes' and 'thing_classes'. "
                f"Generating {len(self._class_names)} generic class names (up to ignore_label={ignore_label})."
            )
        # --- End Handling ---

        self._class_offset = meta.class_offset if hasattr(meta, 'class_offset') else 0
        self._num_classes = len(self._class_names)
        self._semseg_loader = meta.semseg_loader if hasattr(meta, 'semseg_loader') else 'PIL'

        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # Check if output directory exists
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        self.reset()

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            # --- Handle different output types --- 
            if isinstance(output, dict) and "sem_seg" in output:
                sem_seg_tensor = output["sem_seg"]
            elif torch.is_tensor(output):
                # Assuming the output is the segmentation tensor itself
                sem_seg_tensor = output
            else:
                self._logger.error(f"Unexpected output type in SemSegEvaluator: {type(output)}. Skipping.")
                continue # Skip this sample
            # --- End Handling ---
                
            # Process the segmentation tensor
            # <<< --- Keep pred as tensor --- >>>
            pred = sem_seg_tensor.argmax(dim=0).to(self._cpu_device)
            # pred = np.array(output_tensor, dtype=int) # Convert later
            # <<< --- End Change --- >>>
            
            gt_file_path = self.input_file_to_gt_file.get(input["file_name"])
            if gt_file_path is None:
                self._logger.warning(f"Could not find ground truth file for input: {input['file_name']}. Skipping sample.")
                continue

            # <<< --- Start Modification --- >>>
            gt_loaded_successfully = False
            gt_original = None
            try:
                with PathManager.open(gt_file_path, "rb") as f:
                    # Load GT as numpy array first for inspection
                    gt_original = np.array(Image.open(f), dtype=np.uint8) # Load as uint8 for inspection
                    gt_loaded_successfully = True
            except FileNotFoundError:
                self._logger.warning(f"Ground truth file not found: {gt_file_path}. Skipping sample associated with {input['file_name']}.")
                continue
            except Exception as e:
                self._logger.error(f"Error loading ground truth image {gt_file_path} for {input['file_name']}: {e}. Skipping sample.")
                continue

            if not gt_loaded_successfully or gt_original is None:
                 # Should not happen if try block succeeded, but as a safeguard
                 self._logger.error(f"Failed to load GT {gt_file_path} despite no exception. Skipping.")
                 continue

            # Check if the mask is binary (0 and 255) - handle potential RGBA
            if gt_original.ndim == 3 and gt_original.shape[2] in [3, 4]:
                 gt_check = gt_original[:, :, 0] # Use first channel for check
            else:
                 gt_check = gt_original

            unique_vals = np.unique(gt_check)
            is_binary_mask = set(unique_vals).issubset({0, 255})

            gt = None # Initialize gt to be filled
            if is_binary_mask:
                self._logger.debug(f"Detected binary mask (0, 255) for {input.get('file_name', 'unknown')}. Processing category ID.")
                try:
                    # Retrieve category_id from the input dict structure
                    # Assuming the structure from register_biomed_datasets.py
                    target_category_id = int(input['grounding_info'][0]['category_id'])

                    # Create the new gt mask, initialized with ignore_label
                    # We need the actual ignore_label value from metadata
                    ignore_value_for_init = self._ignore_label if isinstance(self._ignore_label, int) else 255 # Default if list?
                    gt = np.full(gt_check.shape, ignore_value_for_init, dtype=np.int64)

                    # Set background pixels to 0
                    gt[gt_check == 0] = 0
                    # Set foreground pixels (255) to the target category_id
                    gt[gt_check == 255] = target_category_id

                except (KeyError, IndexError, TypeError, ValueError) as e:
                    self._logger.error(f"Error extracting target category_id for binary mask {input.get('file_name', 'unknown')}: {e}. Input keys: {input.keys()}. Skipping sample.")
                    continue # Skip this sample if category_id cannot be determined
            else:
                 # Not a binary 0/255 mask, assume pixels are category IDs directly
                 # Apply class offset
                 gt = gt_original.astype(np.int64) - self._class_offset
            # <<< --- End Modification --- >>>

            # <<< --- Convert GT to Tensor earlier --- >>>
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt.astype("long")).to(self._cpu_device)
            elif isinstance(gt, torch.Tensor): # Already tensor?
                gt = gt.to(self._cpu_device)
            else:
                logger.error(f"Unsupported GT type after loading: {type(gt)}. Skipping sample.")
                continue
            # <<< --- End Conversion --- >>>

            # Ensure pred and gt have compatible shapes [H, W] for confusion matrix
            gt_shape = gt.shape
            pred_shape = pred.shape

            # Ensure pred and gt have compatible shapes [H, W] for confusion matrix
            compatible_shapes = False
            if gt.ndim == pred.ndim and gt.ndim == 2 and gt_shape == pred_shape:
                 # Case 0: Already compatible [H, W]
                 compatible_shapes = True
            elif gt.ndim == 3 and pred.ndim == 2:
                # Case 1: gt is multi-channel [C, H, W], pred is single-channel [H, W]
                target_spatial_shape = gt_shape[-2:]
                if pred_shape != target_spatial_shape:
                    # logger.warning(
                    #     f"[{input.get('file_name', 'unknown')}] Prediction shape {pred_shape} != GT spatial shape {target_spatial_shape}. "
                    #     f"Resizing prediction."
                    # )
                    try:
                        # <<< --- Ensure pred is tensor for resize --- >>>
                        pred = TF.resize(pred.unsqueeze(0).float(), target_spatial_shape, interpolation=InterpolationMode.NEAREST).squeeze(0).long()
                        pred_shape = pred.shape # Update shape after resize
                    except Exception as e:
                         logger.error(f"Failed to resize prediction: {e}. Skipping item.")
                         continue

                logger.warning(f"[{input.get('file_name', 'unknown')}] Ground truth shape {gt_shape} has multiple channels. Converting to single channel label map using torch.max.")
                try:
                    gt = torch.max(gt, dim=0)[0]
                    gt_shape = gt.shape # Update shape after conversion
                    compatible_shapes = True
                except Exception as e:
                     logger.error(f"Failed to convert multi-channel GT: {e}. Skipping item.")
                     continue

            elif gt.ndim == 2 and pred.ndim == 2 and gt_shape != pred_shape:
                # Case 2: Both are single-channel [H, W] but spatial shapes differ
                #  logger.warning(
                #      f"[{input.get('file_name', 'unknown')}] Prediction shape {pred_shape} != ground truth shape {gt_shape}. "
                #      f"Resizing prediction to match GT shape."
                #  )
                 try:
                     # Use gt_shape directly as size for TF.resize expects (h, w)
                     # <<< --- Ensure pred is tensor for resize --- >>>
                     pred = TF.resize(pred.unsqueeze(0).float(), gt_shape, interpolation=InterpolationMode.NEAREST).squeeze(0).long()
                     pred_shape = pred.shape # Update shape after resize
                     compatible_shapes = True
                 except Exception as e:
                      logger.error(f"Failed to resize prediction: {e}. Skipping item.")
                      continue

            # Check compatibility before proceeding
            if not compatible_shapes or gt.shape != pred.shape:
                 logger.error(f"[{input.get('file_name', 'unknown')}] Failed to make shapes compatible: final pred {pred.shape}, final gt {gt.shape}. Skipping confusion matrix update.")
                 continue # Skip to next item in batch

            # <<< --- Convert to numpy and clip/ignore just before bincount --- >>>
            pred_np = pred.cpu().numpy().astype(int)
            gt_np = gt.cpu().numpy().astype(int)

            # <<< --- Handle ignore_label on NumPy array --- >>>
            if isinstance(self._ignore_label, int):
                # Ensure gt is writable before modification
                if not gt_np.flags.writeable:
                    gt_np = gt_np.copy()
                ignore_val = self._ignore_label - self._class_offset if self._ignore_label != 255 else self._num_classes # Map 255 to num_classes index
                gt_np[gt_np == (self._ignore_label - self._class_offset)] = ignore_val
            elif isinstance(self._ignore_label, list):
                 if not gt_np.flags.writeable:
                    gt_np = gt_np.copy()
                 for ignore_l in self._ignore_label:
                     ignore_val = ignore_l - self._class_offset if ignore_l != 255 else self._num_classes
                     gt_np[gt_np == (ignore_l - self._class_offset)] = ignore_val
            # <<< --- End ignore_label handling --- >>>

            # Clip gt to valid range [0, num_classes]
            gt_np = np.clip(gt_np, 0, self._num_classes)
            # Clip predictions to valid range [0, num_classes]
            pred_np = np.clip(pred_np, 0, self._num_classes)

            conf_mat = np.bincount(
                (self._num_classes + 1) * pred_np.reshape(-1) + gt_np.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            self._conf_matrix += conf_mat
            
            # <<< --- Use NumPy array for encoding --- >>>
            self._predictions.extend(self.encode_json_sem_seg(pred_np, input["file_name"]))
            # <<< --- End Change --- >>>

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return
            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list
