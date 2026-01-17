# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy
import multiprocessing
import functools

# ==========================================================
# 【新增补丁】解决 PyTorch 2.6+ 报错 "Weights only load failed"
# ==========================================================
import torch
try:
    # 备份原有的 torch.load
    _original_torch_load = torch.load
    
    # 定义一个新的 load 函数，强制 weights_only=False
    def _patched_torch_load(*args, **kwargs):
        # 如果调用者没有指定 weights_only，我们强制设为 False
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    
    # 用我们的函数覆盖掉官方的
    torch.load = _patched_torch_load
    print("【系统提示】已自动修补 PyTorch 2.6+ 的 checkpoint 加载限制。")
except Exception as e:
    print(f"【系统警告】修补 torch.load 失败，如果遇到加载错误请忽略此警告: {e}")
# ==========================================================

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS, HOOKS
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.hooks import Hook
import random
import cv2
import numpy as np
import mmcv
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData

# Global list to share data between Hook and Main
COLLECTED_RESULTS = []

@HOOKS.register_module()
class DualModalVisCollectorHook(Hook):
    """
    Lightweight Hook to collect data for Dual-Modal Visualization.
    It does NOT draw images. It only collects paths and boxes.
    Drawing happens in main() using multiprocessing.
    """
    def __init__(self, show_num):
        self.show_num = show_num
        self.selected_indices = None
            
    def before_test_epoch(self, runner):
        # Initialize random selection
        dataset = runner.test_loop.dataloader.dataset
        try:
            total_len = len(dataset)
        except:
            # Fallback if len is not available
            total_len = 10000 
            
        if self.show_num >= total_len:
            self.selected_indices = set(range(total_len))
        else:
            self.selected_indices = set(random.sample(range(total_len), self.show_num))
        print(f"\n[DualModalVisCollectorHook] Will collect {len(self.selected_indices)} samples for post-process visualization.")

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # outputs: list of DetDataSample (Predictions)
        # data_batch: dict containing 'data_samples' (list of DetDataSample with GT and Metainfo)
        
        batch_size = len(outputs)
        start_idx = batch_idx * runner.test_loop.dataloader.batch_size
        
        input_data_samples = data_batch['data_samples']
        
        for i, pred_sample in enumerate(outputs):
            global_idx = start_idx + i
            
            if global_idx not in self.selected_indices:
                continue
                
            # Get corresponding input info and GT
            gt_sample = input_data_samples[i]
            
            # Extract necessary data
            # We need:
            # - Image Paths
            # - GT Instances (cpu)
            # - Pred Instances (cpu)
            # - Classes (from dataset in runner)
            
            img_path = gt_sample.metainfo['img_path']
            img_path2 = gt_sample.metainfo.get('img_path2', None)
            
            # Clone instances to CPU to verify they are detached and safe for transport
            gt_instances = gt_sample.gt_instances.cpu().clone()
            pred_instances = pred_sample.pred_instances.cpu().clone()
            
            # Store tuple
            COLLECTED_RESULTS.append({
                'img_path': img_path,
                'img_path2': img_path2,
                'gt_instances': gt_instances,
                'pred_instances': pred_instances,
                'metainfo': gt_sample.metainfo, # Contains flip, scale etc if needed
            })
            
# Worker Function for Multiprocessing
def draw_single_sample(sample_data, show_dir, classes):
    """
    Worker function to draw a single dual-modal sample.
    """
    try:
        img_path = sample_data['img_path']
        img_path2 = sample_data['img_path2']
        gt_instances = sample_data['gt_instances']
        pred_instances = sample_data['pred_instances']
        # metainfo = sample_data['metainfo']
        
        # Instantiate Visualizer locally
        visualizer = DetLocalVisualizer()
        visualizer.dataset_meta = {'classes': classes}
        
        # Read images
        img1 = mmcv.imread(img_path)
        if img_path2:
            img2 = mmcv.imread(img_path2)
        else:
            img2 = np.zeros_like(img1)

        # Resize img2 to match img1 if needed
        if img1.shape != img2.shape:
            img2 = mmcv.imresize(img2, (img1.shape[1], img1.shape[0]))
            
        # Helper method for drawing
        def draw_panel(img, instances, is_gt=True, score_thr=0.3):
            # Workaround: DetLocalVisualizer expects bboxes to be tensor for .sum() check
            # but sometimes gets HorizontalBoxes which lacks .sum()
            if 'bboxes' in instances:
                if hasattr(instances.bboxes, 'tensor'):
                    instances.bboxes = instances.bboxes.tensor
            
            visualizer.set_image(img)
            
            # Construct a dummy DataSample for visualizer
            from mmdet.structures import DetDataSample
            data_sample = DetDataSample()
            data_sample.set_metainfo({'img_shape': img.shape[:2]})
            
            if is_gt:
                data_sample.gt_instances = instances
                visualizer.add_datasample(
                    'draw', img, data_sample=data_sample,
                    draw_gt=True, draw_pred=False, show=False, out_file=None)
            else:
                data_sample.pred_instances = instances
                visualizer.add_datasample(
                    'draw', img, data_sample=data_sample,
                    draw_gt=False, draw_pred=True, pred_score_thr=score_thr, show=False, out_file=None)
            
            return visualizer.get_image()
            
        # Draw 1: Optical GT
        img1_gt = draw_panel(img1, gt_instances, is_gt=True)
        
        # Draw 2: Optical Pred
        img1_pred = draw_panel(img1, pred_instances, is_gt=False, score_thr=0.3)
        
        # Draw 3: SAR GT
        img2_gt = draw_panel(img2, gt_instances, is_gt=True)
        
        # Draw 4: SAR Pred
        img2_pred = draw_panel(img2, pred_instances, is_gt=False, score_thr=0.3)
        
        # Concatenate: 
        # [Opt GT] [Opt Pred]
        # [SAR GT] [SAR Pred]
        
        # Add labels
        font_scale = 1.0
        color = (0, 0, 255)
        thickness = 2
        
        cv2.putText(img1_gt, "Optical GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(img1_pred, "Optical Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(img2_gt, "SAR GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        cv2.putText(img2_pred, "SAR Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        row1 = np.concatenate((img1_gt, img1_pred), axis=1)
        row2 = np.concatenate((img2_gt, img2_pred), axis=1)
        final_canvas = np.concatenate((row1, row2), axis=0)
        
        # Save
        base_name = os.path.basename(img_path)
        out_file = os.path.join(show_dir, base_name)
        mmcv.imwrite(final_canvas, out_file)
        return True
    except Exception as e:
        print(f"Error drawing {sample_data.get('img_path')}: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--show-num',
        type=int,
        default=None,
        help='Randomly visualize N images (testing runs on full dataset)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Configure Visualization Collector Hook
    # By default, trigger_visualization_hook uses args.show_dir to enable drawing.
    # If using show_num, we want to disable standard drawing to avoid performance hit
    # and use our custom post-process visualizer instead.
    custom_show_dir = None
    if args.show_num is not None:
        custom_show_dir = args.show_dir
        # We temporarily unset args.show_dir so trigger_visualization_hook doesn't enable default drawing
        # But we keep it in custom_show_dir for our use.
        # However, trigger_visualization_hook calculates work_dir/timestamp/show_dir if show_dir is NOT None.
        # If we set it to None, it won't calculate that path.
        # So we let it run, then FORCE disable it.
        pass

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)
    
    # NOW we override the visualization hook if show_num is active
    if args.show_num is not None and custom_show_dir is not None:
        # 1. Disable default visualization drawing
        if 'visualization' in cfg.default_hooks:
            cfg.default_hooks.visualization.draw = False
        
        # Also check if there's a custom visualization hook added by trigger_visualization_hook
        # Usually it modifies default_hooks.visualization
        
        # 2. Add our Collector Hook
        custom_hook_cfg = dict(
            type='DualModalVisCollectorHook',
            show_num=args.show_num,
            priority='LOWEST'
        )
        if cfg.get('custom_hooks', None) is None:
            cfg.custom_hooks = []
        cfg.custom_hooks.append(custom_hook_cfg)
        
        # Ensure show_dir exists (use the one potentially modified by user or calculate it ourselves if needed)
        # Note: args.show_dir might be relative.
        if not os.path.exists(custom_show_dir):
            os.makedirs(custom_show_dir)
            
        print(f"[Config] Disabled standard visualization. Using custom collector for {args.show_num} samples.")
        print(f"[Config] Custom visualization output: {custom_show_dir}")

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()
    
    # ==========================================================
    # Post-Processing: Parallel Visualization
    # ==========================================================
    if args.show_num is not None and args.show_dir is not None and len(COLLECTED_RESULTS) > 0:
        print(f"\n[Post-Process] Starting visualization of {len(COLLECTED_RESULTS)} samples...")
        print(f"[Post-Process] Saving to: {args.show_dir}")
        print(f"[Post-Process] Usage: Using multiprocessing with available CPUs.")
        
        # Determine classes
        classes = runner.test_loop.dataloader.dataset.metainfo.get('classes', None)
        
        # Prepare arguments for worker
        # We need to use partial to pass constant args
        worker_func = functools.partial(draw_single_sample, show_dir=args.show_dir, classes=classes)
        
        # Use Pool
        # Note: 'spawn' context might be needed for some libs, but 'fork' is default on Linux and faster.
        # However, reusing MM/Torch objects across processes can be tricky.
        # We converted instances to CPU Tensors, which should pickle fine.
        
        num_workers = min(multiprocessing.cpu_count(), 32)
        print(f"[Post-Process] Spawning {num_workers} workers.")
        
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                # chunksize can be adjusted
                results = pool.map(worker_func, COLLECTED_RESULTS, chunksize=10)
        except Exception as e:
             print(f"Post processing failed: {e}")
             results = []
            
        success_count = sum(results)
        print(f"[Post-Process] Visualization complete. Success: {success_count}/{len(COLLECTED_RESULTS)}")

if __name__ == '__main__':
    # Set start method to spawn to avoid CUDA initialization issues in children if any
    # But usually 'fork' is fine if we don't use CUDA in children.
    # We explicitly move tensors to CPU before collection.
    # multiprocessing.set_start_method('spawn', force=True)
    main()
