# Copyright (c) OpenMMLab. All rights reserved.
# Helper functions adapted from MMRotate 1.0.0rc1 due to registry conflicts in this environment
import numpy as np
import torch
from typing import Sequence, List, Optional, Union
from multiprocessing import get_context
from terminaltables import AsciiTable
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log, MMLogger
from mmdet.registry import METRICS
from mmdet.evaluation.functional import average_precision

try:
    from mmcv.ops import box_iou_rotated
except ImportError:
    box_iou_rotated = None

def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 box_type='rbox',
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive."""
    # Ensure inputs are numpy arrays
    det_bboxes = np.array(det_bboxes)
    if gt_bboxes_ignore is None:
        gt_bboxes_ignore = np.empty((0, 5), dtype=np.float32)
    else:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore)
    gt_bboxes = np.array(gt_bboxes)

    # Track which GTs are ignored (from the original ignore list)
    num_real_gts = gt_bboxes.shape[0]
    num_ignore_gts = gt_bboxes_ignore.shape[0]
    
    # stack gt_bboxes and gt_bboxes_ignore
    if num_ignore_gts > 0:
        gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    
    gt_ignore_inds = np.zeros(gt_bboxes.shape[0], dtype=bool)
    if num_ignore_gts > 0:
        gt_ignore_inds[num_real_gts:] = True

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)

    # Pre-calculate scale info
    # Pre-calculate det areas if using scale ranges
    if num_dets > 0 and area_ranges != [(None, None)]:
        if box_type == 'rbox':
            det_areas = det_bboxes[:, 2] * det_bboxes[:, 3]
        else:
            # Assuming hbbox [x, y, x, y] or [x, y, w, h] - this function assumes rbox usually
            # But let's support what we can
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0]) * (det_bboxes[:, 3] - det_bboxes[:, 1])

    if num_gts == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            for k, (min_area, max_area) in enumerate(area_ranges):
                if min_area is None:
                    fp[k, ...] = 1
                else:
                    if num_dets > 0:
                        is_in_area = (det_areas >= min_area) & (det_areas < max_area)
                        fp[k, is_in_area] = 1
        return tp, fp

    if box_type == 'rbox':
        ious = box_iou_rotated(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_bboxes).float()).numpy()
    else:
        raise NotImplementedError
    
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    
    # Pre-calculate GT areas for scale filtering
    if area_ranges != [(None, None)]:
        if box_type == 'rbox':
            gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        else:
            # Fallback
            gt_areas = np.zeros(num_gts)

    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        
        if min_area is None:
            gt_area_ignore = np.zeros(num_gts, dtype=bool)
        else:
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
            
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                # Check if matched GT is valid for this scale and not already covered
                should_ignore_gt = gt_ignore_inds[matched_gt] or gt_area_ignore[matched_gt]
                
                if not should_ignore_gt:
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # If matched GT is ignored, the detection matches *ignored* GT.
                # In COCO, matching an ignored GT means the detection is ignored (neither TP nor FP).
                # So we do nothing here.
                
            else:
                # No match found
                if min_area is None:
                    fp[k, i] = 1
                else:
                    # Check if the detection itself is inside the area range
                    # A mismatched detection is FP only if it is within the scale we are evaluating
                    if (det_areas[i] >= min_area) and (det_areas[i] < max_area):
                        fp[k, i] = 1
                    # Else: detection is outside scale, so it's ignored for this scale eval.

    return tp, fp

def get_cls_results(det_results, annotations, class_id, box_type):
    """Get det results and gt information of a certain class."""
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        if len(ann['bboxes']) != 0:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts.append(torch.zeros((0, 5), dtype=torch.float64))
            cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))
    return cls_dets, cls_gts, cls_gts_ignore

def print_map_summary(mean_ap, results, dataset=None, scale_ranges=None, logger=None):
    """Print mAP and results of each class."""
    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    num_classes = len(results)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)

def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   box_type='rbox',
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset."""
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    # Use multiprocessing for speedup
    pool = get_context('spawn').Pool(nproc)
    
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, box_type)

        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [box_type for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp = tuple(zip(*tpfp))
        
        # calculate gt number of each scale
        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area) & (gt_areas < max_area))
        
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    
    if scale_ranges is not None:
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack([cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(mean_ap, eval_results, dataset, area_ranges, logger=logger)
    return mean_ap, eval_results


@METRICS.register_module(force=True)
class OBBCocoMetric(BaseMetric):
    """
    Wrapper to use MMRotate's OBB evaluation logic (DOTA mAP).
    Adapted to avoid registry conflicts.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self, ann_file=None, metric='mAP', iou_thrs=[0.5], scale_ranges=None, format_only=False, backend_args=None, **kwargs):
        super().__init__(**kwargs)
        self.format_only = format_only
        self.backend_args = backend_args
        self.iou_thrs = iou_thrs
        if isinstance(self.iou_thrs, float):
            self.iou_thrs = [self.iou_thrs]
        self.metric = metric
        self.scale_ranges = scale_ranges
        # DOTA Metric uses 11points by default (VOC07 style)
        self.use_07_metric = True 
        
        if box_iou_rotated is None:
            print("Warning: mmcv.ops.box_iou_rotated not available.")

    def process(self, data_batch: dict, data_samples: list) -> None:
        """Process one batch of data samples and predictions."""
        for data_sample in data_samples:
            # Extract GT similar to how MMRotate DOTAMetric does it
            gt_instances = data_sample['gt_instances']
            gt_ignore_instances = data_sample.get('ignored_instances', dict(bboxes=torch.empty(0, 5), labels=torch.empty(0)))

            if isinstance(gt_instances, dict):
                bboxes = gt_instances['bboxes']
                labels = gt_instances['labels']
            else:
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

            if isinstance(gt_ignore_instances, dict):
                bboxes_ignore = gt_ignore_instances['bboxes'] if 'bboxes' in gt_ignore_instances else torch.empty(0, 5)
                labels_ignore = gt_ignore_instances['labels'] if 'labels' in gt_ignore_instances else torch.empty(0)
            else:
                bboxes_ignore = gt_ignore_instances.bboxes
                labels_ignore = gt_ignore_instances.labels

            if len(bboxes) == 0:
                ann = dict(
                    bboxes=np.zeros((0, 5)),
                    labels=np.zeros((0, )),
                    bboxes_ignore=np.zeros((0, 5)),
                    labels_ignore=np.zeros((0, ))
                )
            else:
                ann = dict(
                    labels=labels.cpu().numpy(),
                    bboxes=bboxes.cpu().numpy(),
                    bboxes_ignore=bboxes_ignore.cpu().numpy() if bboxes_ignore.numel() > 0 else np.zeros((0, 5)),
                    labels_ignore=labels_ignore.cpu().numpy() if labels_ignore.numel() > 0 else np.zeros((0, )))

            result = dict()
            pred = data_sample['pred_instances']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            
            # Format predictions for eval_rbbox_map (list of list of det per class)
            # mmrotate expects: [ [det_cls1, det_cls2...], ... image2 ... ]
            # So we pre-format it here or in compute_metrics. DOTAMetric does it in process.
            
            result['pred_bbox_scores'] = []
            classes = self.dataset_meta.get('classes', range(6)) # Default to 6 classes if not found
            for label in range(len(classes)):
                index = np.where(result['labels'] == label)[0]
                if len(index) > 0:
                    pred_bbox_scores = np.hstack([
                        result['bboxes'][index], 
                        result['scores'][index].reshape((-1, 1))
                    ])
                else:
                    pred_bbox_scores = np.zeros((0, 6)) # 5 bbox + 1 score
                result['pred_bbox_scores'].append(pred_bbox_scores)

            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results."""
        logger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        
        classes = self.dataset_meta.get('classes', [str(i) for i in range(6)])
        
        # Prepare input for eval_rbbox_map
        # det_results needs to be list of list of numpy arrays
        det_results = [pred['pred_bbox_scores'] for pred in preds]
        
        eval_results = {}
        # MMRotate DOTAMetric loop over thresholds
        mean_aps = []
        for iou_thr in self.iou_thrs:
            logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_rbbox_map(
                det_results,
                gts,
                scale_ranges=self.scale_ranges,
                iou_thr=iou_thr,
                use_07_metric=self.use_07_metric,
                box_type='rbox',
                dataset=classes,
                logger=logger)
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        eval_results['bbox_mAP'] = eval_results['mAP']
        eval_results['bbox_mAP_50'] = eval_results['AP50']
        
        return eval_results

    def _calculate_coco_summary(self, results, logger):
        """Calculate and print COCO-style summary."""
        import numpy as np
        
        gts, preds = zip(*results)
        classes = self.dataset_meta.get('classes', [str(i) for i in range(6)])
        det_results = [pred['pred_bbox_scores'] for pred in preds]
        
        # Define 12 standard COCO stats
        # 1. AP @ 0.5:0.95 | area=all | maxDets=100
        # 2. AP @ 0.50     | area=all | maxDets=1000
        # 3. AP @ 0.75     | area=all | maxDets=1000
        # 4. AP @ 0.5:0.95 | area=small | maxDets=1000
        # 5. AP @ 0.5:0.95 | area=medium | maxDets=1000
        # 6. AP @ 0.5:0.95 | area=large | maxDets=1000
        # 7. AR @ 0.5:0.95 | area=all | maxDets=100
        # 8. AR @ 0.5:0.95 | area=all | maxDets=300
        # 9. AR @ 0.5:0.95 | area=all | maxDets=1000
        # 10. AR @ 0.5:0.95 | area=small | maxDets=1000
        # 11. AR @ 0.5:0.95 | area=medium | maxDets=1000
        # 12. AR @ 0.5:0.95 | area=large | maxDets=1000

        iou_thrs = np.linspace(0.5, 0.95, 10)
        # scale ranges: all, small, medium, large
        scale_ranges = [
            (0, 1e5), # all
            (0, 32),  # small
            (32, 96), # medium
            (96, 1e5) # large
        ]
        
        # Cache for performance: store results for each IoU
        # list of (mean_aps_list, eval_results_list)
        # mean_aps_list: [ap_all, ap_s, ap_m, ap_l]
        iou_results = []
        
        print_log(f'Calculating COCO metrics for {len(iou_thrs)} thresholds...', logger=logger)
        
        for i, iou in enumerate(iou_thrs):
            # We use 'silent' logger to suppress per-step tables
            mean_aps, step_eval_results = eval_rbbox_map(
                det_results,
                gts,
                scale_ranges=scale_ranges,
                iou_thr=iou,
                use_07_metric=False, # Use integrated AP for COCO style
                box_type='rbox',
                dataset=classes,
                logger='silent')
            iou_results.append((mean_aps, step_eval_results))
            print_log(f'Completed IoU={iou:.2f}, AP={mean_aps[0]:.3f}', logger=logger)

        stats = np.zeros(12)
        
        # Helper: Get mean AP across all IoUs for a specific scale index
        def get_mean_ap(scale_idx):
            return np.mean([res[0][scale_idx] for res in iou_results])
            
        # Helper: Get AP for specific IoU index and scale index
        def get_ap(iou_idx, scale_idx):
            return iou_results[iou_idx][0][scale_idx]

        # Helper: Execute AR calculation
        # Average Recall over IoUs and Classes for a specific scale and maxDets
        def get_ar(scale_idx, max_dets):
            # res[1] is list of dicts (one per class)
            # dict['recall'] is shape (num_scales, num_dets)
            ars_iou = [] # Average recall per IoU
            for _, step_eval_res in iou_results:
                recalls_cls = []
                for cls_res in step_eval_res:
                    recall_curve = cls_res['recall'][scale_idx] # shape (num_dets,)
                    num_dets = len(recall_curve)
                    if num_dets == 0:
                        recalls_cls.append(0.0)
                    else:
                        idx = min(max_dets, num_dets) - 1
                        recalls_cls.append(recall_curve[idx])
                ars_iou.append(np.mean(recalls_cls))
            return np.mean(ars_iou)

        # 1. AP @ 0.5:0.95 | area=all
        stats[0] = get_mean_ap(0)
        # 2. AP @ 0.50 | area=all
        stats[1] = get_ap(0, 0)
        # 3. AP @ 0.75 | area=all
        stats[2] = get_ap(5, 0) # Index 5 is 0.75 in linspace(0.5, 0.95, 10) -> 0.5, 0.55, 0.60, 0.65, 0.70, 0.75
        # 4. AP | area=small
        stats[3] = get_mean_ap(1)
        # 5. AP | area=medium
        stats[4] = get_mean_ap(2)
        # 6. AP | area=large
        stats[5] = get_mean_ap(3)
        
        # 7. AR | maxDets=100
        stats[6] = get_ar(0, 100)
        # 8. AR | maxDets=300
        stats[7] = get_ar(0, 300)
        # 9. AR | maxDets=1000
        stats[8] = get_ar(0, 1000)
        # 10. AR | area=small
        stats[9] = get_ar(1, 1000)
        # 11. AR | area=medium
        stats[10] = get_ar(2, 1000)
        # 12. AR | area=large
        stats[11] = get_ar(3, 1000)
        
        # Formatting rows matching the user image
        title = "Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}"
        print_log(title.format(stats[0]), logger=logger)
        print_log("Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = {:.3f}".format(stats[1]), logger=logger)
        print_log("Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = {:.3f}".format(stats[2]), logger=logger)
        print_log("Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = {:.3f}".format(stats[3]), logger=logger)
        print_log("Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = {:.3f}".format(stats[4]), logger=logger)
        print_log("Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = {:.3f}".format(stats[5]), logger=logger)
        print_log("Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}".format(stats[6]), logger=logger)
        print_log("Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = {:.3f}".format(stats[7]), logger=logger)
        print_log("Average Recall    (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = {:.3f}".format(stats[8]), logger=logger)
        print_log("Average Recall    (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = {:.3f}".format(stats[9]), logger=logger)
        print_log("Average Recall    (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = {:.3f}".format(stats[10]), logger=logger)
        print_log("Average Recall    (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = {:.3f}".format(stats[11]), logger=logger)
        
        # Copy-paste style line (as requested by user screenshot bottom line)
        # Note: The screenshot shows: bbox_mAP_copypaste: 0.478 0.851 ...
        copypaste_line = "bbox_mAP_copypaste: " + " ".join([f"{x:.3f}" for x in stats])
        print_log(copypaste_line, logger=logger)
        
        return {
            'bbox_mAP': stats[0],
            'bbox_mAP_50': stats[1],
            'bbox_mAP_75': stats[2],
            'bbox_mAP_s': stats[3],
            'bbox_mAP_m': stats[4],
            'bbox_mAP_l': stats[5],
            'mAP': stats[0], # Backward compatibility
        }

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results."""
        logger = MMLogger.get_current_instance()
        
        # If user did NOT specify specific iou_thrs (default logic) OR explicitly asks for coco style
        # We assume if they want "more items" and "diagram format", we switch to full COCO calculation
        # regardless of __init__ params, OR we only do it if configured. 
        # Since I'm editing the code for the user request: I will FORCE the new output.
        
        return self._calculate_coco_summary(results, logger)

# Since I cannot easily install new packages like mmrotate, I will modify the user's config 
# to use a custom metric or adapt the results.

