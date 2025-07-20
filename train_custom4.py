import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import sys
import shutil
from datetime import datetime

# ---- Logger class to capture terminal output to a file ----
class Logger:
    """Redirects print statements to both the console and a file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile = open(filepath, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

# ---- YOLOv5 IMPORT FIX ----
# Ensure the script can find the YOLOv5 repository
yolov5_root_path = os.environ.get('YOLOV5_ROOT', '/content/yolov5')
if Path(yolov5_root_path).is_dir() and str(yolov5_root_path) not in sys.path:
    print(f"Adding YOLOv5 root to system path: {yolov5_root_path}")
    sys.path.append(str(yolov5_root_path))

# Import YOLOv5 utilities
try:
    from train import train
    from utils.callbacks import Callbacks
    from models.yolo import Model
    from utils.general import check_yaml
    from models.experimental import attempt_load as yolo_attempt_load
except ImportError as e:
    print("FATAL: Could not import YOLOv5 utilities. This script requires a YOLOv5 environment.")
    print(f"Please ensure the YOLOv5 repository is cloned and its path is correct.")
    print(f"ImportError: {e}")
    sys.exit(1)

# ---- UPDATED: Per-Epoch Drive Sync Callback ----
class DriveSyncCallback(Callbacks):
    """
    A YOLOv5 callback to sync the entire run directory to a destination (e.g., Google Drive)
    after every single training epoch. This replaces the destination directory on each sync.

    NOTE: This performs a full directory copy on each epoch, which can be I/O intensive
    and may slow down training, especially in environments like Google Colab.
    """
    def __init__(self, local_run_dir, drive_run_dir):
        super().__init__()
        self.local_run_dir = Path(local_run_dir)
        self.drive_run_dir = Path(drive_run_dir)
        
        # We only need to ensure the parent of the destination directory exists.
        self.drive_run_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Drive Sync enabled. After each epoch, '{self.local_run_dir}' will be fully synced to '{self.drive_run_dir}'.")

    def on_fit_epoch_end(self, epoch, results, best, fi, last, stop, best_fitness, early_stop_epoch):
        """Runs at the end of each epoch to sync the entire run folder."""
        # The training run directory is created by YOLOv5 on the first batch, not before.
        # So we must check if it exists before trying to copy it.
        if not self.local_run_dir.is_dir():
            print(f"INFO: Source directory '{self.local_run_dir}' not yet created. Skipping sync for epoch {epoch}.")
            return

        print(f"\n--- Epoch {epoch} complete. Syncing all files and subfolders to {self.drive_run_dir} ---")
        try:
            # For a clean sync, first remove the old destination directory if it exists.
            if self.drive_run_dir.exists():
                shutil.rmtree(str(self.drive_run_dir))
            
            # Copy the entire local run directory to the destination.
            shutil.copytree(str(self.local_run_dir), str(self.drive_run_dir))
            print(f"Successfully synced '{self.local_run_dir}' to '{self.drive_run_dir}'.")
        except Exception as e:
            print(f"WARNING: An error occurred during the sync to Drive: {e}")

def get_model_weights_as_vector(model):
    """Extracts model weights into a single flat vector."""
    return np.concatenate([p.data.clone().cpu().numpy().flatten() for p in model.parameters() if p.requires_grad])

def set_model_weights_from_vector(model, weights_vector):
    """Sets model weights from a single flat vector."""
    pointer = 0
    for param in model.parameters():
        if param.requires_grad:
            num_param = param.numel()
            param.data = torch.from_numpy(weights_vector[pointer:pointer + num_param]).view_as(param).to(param.device)
            pointer += num_param

def zero_top_weights(model, percentile=90):
    """Zeros out weights with the largest magnitudes."""
    weights = get_model_weights_as_vector(model)
    threshold = np.percentile(np.abs(weights), percentile)
    weights[np.abs(weights) > threshold] = 0
    set_model_weights_from_vector(model, weights)
    print(f"Pruned weights above the {percentile}th percentile (magnitude > {threshold:.4f})")

def save_checkpoint(model, path, epoch=-1, optimizer=None):
    """Saves a model checkpoint in the format expected by YOLOv5 train.py."""
    ckpt = {
        'epoch': epoch,
        'best_fitness': None,
        'model': model,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'wandb_id': None,
        'date': datetime.now().isoformat()
    }
    torch.save(ckpt, path)
    print(f"Saved YOLOv5-compatible checkpoint to {path}")

def main(opt):
    # --- SETUP DESCRIPTIVE NAMES AND PATHS ---
    date_str = datetime.now().strftime('%Y-%m-%d')
    model_name = Path(opt.initial_weights).stem
    prune_percent = 100 - opt.prune_keep_percent
    prune_percent_str = f'{prune_percent:g}'

    descriptive_name = (f"{date_str}_{model_name}_initial{opt.pruning_epoch}e_pruned{prune_percent_str}pct_"
                        f"final{opt.total_epochs}e")
    
    # This is the root directory for all local outputs of this specific run
    local_project_dir = Path(opt.project) / descriptive_name
    local_project_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = local_project_dir / f"{descriptive_name}.log"
    pruned_model_path = local_project_dir / f'{descriptive_name}_pruned.pt'
    averaged_model_path = local_project_dir / f'{descriptive_name}_averaged.pt'
    
    initial_train_name = f'initial_train'
    final_train_name = f'final_train'
    initial_train_save_dir = local_project_dir / initial_train_name
    final_train_save_dir = local_project_dir / final_train_name
    
    drive_base_dir = None
    if opt.save_to_drive:
        # This is the single destination folder on Google Drive for this run
        drive_base_dir = Path('/content/drive/My Drive') / opt.drive_folder_path / descriptive_name
        print(f"Google Drive sync is enabled. Base path: {drive_base_dir}")

    original_stdout = sys.stdout
    sys.stdout = Logger(str(log_file_path))
    
    print(f"Starting run: {descriptive_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- SETUP CALLBACKS ---
    # Create one callback instance to use for all training stages.
    callbacks = Callbacks()  # Default non-syncing callbacks
    if opt.save_to_drive and drive_base_dir:
        callbacks = DriveSyncCallback(local_project_dir, drive_base_dir)

    try:
        # Step 1: Initial Training
        if not opt.resume_run or not (initial_train_save_dir / 'weights' / 'best.pt').exists():
            print(f"\n--- Step 1: Initial training for {opt.pruning_epoch} epochs ---")
            train_opt_dict = {
                'weights': opt.initial_weights, 'cfg': opt.cfg, 'data': opt.data,
                'hyp': 'data/hyps/hyp.scratch-low.yaml', 'epochs': opt.pruning_epoch,
                'batch_size': opt.batch_size, 'imgsz': opt.img_size,
                'project': str(local_project_dir), 'name': initial_train_name, 'exist_ok': True,
                'device': str(device), 'cache': opt.cache, 'workers': opt.workers,
                'rect': True, 'resume': False, 'image_weights': False, 'nosave': False,
                'noval': False, 'noautoanchor': False, 'noplots': False, 'evolve': None,
                'bucket': '', 'multi_scale': False, 'single_cls': False, 'optimizer': 'SGD',
                'sync_bn': False, 'quad': False, 'cos_lr': False, 'label_smoothing': 0.0,
                'patience': 100, 'freeze': [0], 'save_period': -1, 'seed': 0, 'local_rank': -1,
                'entity': None, 'upload_dataset': False, 'bbox_interval': -1, 'artifact_alias': "latest"
            }
            train_opt_ns = argparse.Namespace(**train_opt_dict)
            train_opt_ns.save_dir = str(initial_train_save_dir)

            train(train_opt_ns.hyp, train_opt_ns, device, callbacks)
            print(f"Initial training complete. Results saved to {initial_train_save_dir}")
        else:
            print(f"\n--- SKIPPING Step 1: Found completed initial training results in {initial_train_save_dir} ---")

        # Step 2: Prune the initially trained model
        best_initial_weights_path = initial_train_save_dir / 'weights' / 'best.pt'
        if not opt.resume_run or not pruned_model_path.exists():
            print("\n--- Step 2: Pruning the initially trained model ---")
            initial_ckpt = torch.load(best_initial_weights_path, map_location=device, weights_only=False) # <-- FIX
            model = initial_ckpt['model'].float()
            for param in model.parameters():
                param.requires_grad = True
            zero_top_weights(model, percentile=opt.prune_keep_percent)
            save_checkpoint(model, pruned_model_path)
        else:
            print(f"\n--- SKIPPING Step 2: Found existing pruned model at {pruned_model_path} ---")

        # Step 3: Average the pre-pruned and post-pruned models
        if not opt.resume_run or not averaged_model_path.exists():
            print("\n--- Step 3: Averaging pre-pruned and post-pruned weights ---")
            initial_ckpt = torch.load(best_initial_weights_path, map_location=device, weights_only=False) # <-- FIX
            pruned_ckpt = torch.load(pruned_model_path, map_location=device, weights_only=False) # <-- FIX
            avg_model = initial_ckpt['model'].float()
            initial_state_dict = avg_model.state_dict()
            pruned_state_dict = pruned_ckpt['model'].state_dict()
            for key in initial_state_dict:
                if key in pruned_state_dict and initial_state_dict[key].dtype.is_floating_point:
                    initial_state_dict[key].data = (initial_state_dict[key].data + pruned_state_dict[key].data) / 2.0
            avg_model.load_state_dict(initial_state_dict)
            save_checkpoint(avg_model, averaged_model_path)
        else:
            print(f"\n--- SKIPPING Step 3: Found existing averaged model at {averaged_model_path} ---")

        # Step 4: Final Training
        remaining_epochs = opt.total_epochs - opt.pruning_epoch
        if remaining_epochs > 0 and (not opt.resume_run or not (final_train_save_dir / 'weights' / 'best.pt').exists()):
            print(f"\n--- Step 4: Final training for remaining {remaining_epochs} epochs ---")
            final_train_opt_dict = {
                'weights': str(averaged_model_path), 'cfg': opt.cfg, 'data': opt.data,
                'hyp': 'data/hyps/hyp.scratch-low.yaml', 'epochs': remaining_epochs,
                'batch_size': opt.batch_size, 'imgsz': opt.img_size,
                'project': str(local_project_dir), 'name': final_train_name, 'exist_ok': True,
                'device': str(device), 'cache': opt.cache, 'workers': opt.workers,
                'rect': True, 'resume': False, 'image_weights': False, 'nosave': False,
                'noval': False, 'noautoanchor': False, 'noplots': False, 'evolve': None,
                'bucket': '', 'multi_scale': False, 'single_cls': False, 'optimizer': 'SGD',
                'sync_bn': False, 'quad': False, 'cos_lr': False, 'label_smoothing': 0.0,
                'patience': 100, 'freeze': [0], 'save_period': -1, 'seed': 0, 'local_rank': -1,
                'entity': None, 'upload_dataset': False, 'bbox_interval': -1, 'artifact_alias': "latest"
            }
            final_train_opt_ns = argparse.Namespace(**final_train_opt_dict)
            final_train_opt_ns.save_dir = str(final_train_save_dir)

            train(final_train_opt_ns.hyp, final_train_opt_ns, device, callbacks)
            print(f"Final training complete. Results saved to {final_train_save_dir}")
        else:
             print(f"\n--- SKIPPING Step 4 or No remaining epochs ---")

    finally:
        # Restore standard output
        if sys.stdout != original_stdout:
            sys.stdout.logfile.close()
            sys.stdout = original_stdout
        
        print(f"\nTerminal output logging complete. Saved to {log_file_path}")
        print("\nProcess finished successfully. ✨")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 Multi-Stage Training and Pruning")
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--initial-weights', type=str, default='yolov5n.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--project', default='runs/custom_train', help='Base local directory for all runs')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    parser.add_argument('--pruning-epoch', type=int, default=10, help='Epochs for initial training BEFORE pruning.')
    parser.add_argument('--total-epochs', type=int, default=50, help='Total COMBINED epochs.')
    parser.add_argument('--prune-keep-percent', type=float, default=90.0, help='Percent of weights to KEEP after pruning.')
    parser.add_argument('--resume-run', action='store_true', help='Resume from the last successfully completed stage.')
    parser.add_argument('--save-to-drive', action='store_true', help='Save all results to Google Drive.')
    parser.add_argument('--drive-folder-path', type=str, default='YOLOv5_Runs', help='Base Google Drive folder for all runs')

    opt = parser.parse_args()

    if opt.total_epochs <= opt.pruning_epoch:
        raise ValueError("Error: --total-epochs must be greater than --pruning-epoch.")
        
    main(opt)
