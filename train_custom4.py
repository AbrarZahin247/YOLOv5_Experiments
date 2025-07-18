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

# ---- OPTIMIZED DYNAMIC DRIVE SAVING CALLBACK ----
class DriveSyncCallback(Callbacks):
    """
    An optimized YOLOv5 callback to save files to Google Drive without blocking training.
    """
    def __init__(self, local_run_dir, drive_run_dir, save_period=10):
        super().__init__()
        self.local_run_dir = Path(local_run_dir)
        self.drive_run_dir = Path(drive_run_dir)
        self.weights_dir = self.local_run_dir / 'weights'
        self.drive_weights_dir = self.drive_run_dir / 'weights'
        self.save_period = save_period
        
        print(f"Drive Sync enabled. Checkpoints will be saved to: {self.drive_weights_dir}")
        self.drive_weights_dir.mkdir(parents=True, exist_ok=True)
        
    def on_fit_epoch_end(self, epoch, results, best, fi, last, stop, best_fitness, early_stop_epoch):
        """Runs at the end of each epoch to sync files intelligently."""
        if best:
            local_best_file = self.weights_dir / 'best.pt'
            if local_best_file.exists():
                try:
                    shutil.copy2(str(local_best_file), self.drive_weights_dir)
                except Exception as e:
                    print(f"WARNING: Failed to copy best.pt to Drive: {e}")

        if epoch > 0 and epoch % self.save_period == 0:
            for filename in ['last.pt', 'results.csv']:
                local_file = self.local_run_dir / filename if filename.endswith('csv') else self.weights_dir / filename
                drive_dest = self.drive_run_dir if filename.endswith('csv') else self.drive_weights_dir
                if local_file.exists():
                    try:
                        shutil.copy2(str(local_file), drive_dest)
                    except Exception as e:
                        print(f"WARNING: Failed to copy {filename} to Drive: {e}")

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
    
    os.makedirs(opt.project, exist_ok=True)
    log_file_path = Path(opt.project) / f"{descriptive_name}.log"
    pruned_model_path = Path(opt.project) / f'{descriptive_name}_pruned.pt'
    averaged_model_path = Path(opt.project) / f'{descriptive_name}_averaged.pt'
    
    initial_train_name = f'initial_train_{descriptive_name}'
    final_train_name = f'final_train_{descriptive_name}'
    initial_train_save_dir = Path(opt.project) / initial_train_name
    final_train_save_dir = Path(opt.project) / final_train_name
    
    drive_base_dir = None
    if opt.save_to_drive:
        drive_base_dir = Path('/content/drive/My Drive') / opt.drive_folder_path / descriptive_name
        print(f"Google Drive sync is enabled. Base path: {drive_base_dir}")

    original_stdout = sys.stdout
    sys.stdout = Logger(str(log_file_path))
    
    print(f"Starting run: {descriptive_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Step 1: Initial Training
        if not opt.resume_run or not (initial_train_save_dir / 'weights' / 'best.pt').exists():
            print(f"\n--- Step 1: Initial training for {opt.pruning_epoch} epochs ---")
            callbacks = DriveSyncCallback(initial_train_save_dir, drive_base_dir / initial_train_name) if drive_base_dir else Callbacks()
            
            train_opt_dict = {
                'weights': opt.initial_weights, 'cfg': opt.cfg, 'data': opt.data,
                'hyp': 'data/hyps/hyp.scratch-low.yaml', 'epochs': opt.pruning_epoch,
                'batch_size': opt.batch_size, 'imgsz': opt.img_size,
                'project': opt.project, 'name': initial_train_name, 'exist_ok': True,
                'device': str(device), 'cache': opt.cache, 'workers': opt.workers,
                'rect': True, 'resume': False, 'image_weights': False, 'nosave': False,
                'noval': False, 'noautoanchor': False, 'noplots': False, 'evolve': None,
                'bucket': '', 'multi_scale': False, 'single_cls': False, 'optimizer': 'SGD',
                'sync_bn': False, 'quad': False, 'cos_lr': False, 'label_smoothing': 0.0,
                'patience': 100, 'freeze': [0], 'save_period': -1, 'seed': 0, 'local_rank': -1,
                'entity': None, 'upload_dataset': False, 'bbox_interval': -1, 'artifact_alias': "latest"
            }
            # *** FIX: Call train() with the correct 4 arguments ***
            train(
                train_opt_dict['hyp'], 
                argparse.Namespace(**train_opt_dict), 
                device, 
                callbacks
            )
            print(f"Initial training complete. Results saved to {initial_train_save_dir}")
        else:
            print(f"\n--- SKIPPING Step 1: Found completed initial training results in {initial_train_save_dir} ---")

        # Step 2: Prune the initially trained model
        best_initial_weights_path = initial_train_save_dir / 'weights' / 'best.pt'
        if not opt.resume_run or not pruned_model_path.exists():
            print("\n--- Step 2: Pruning the initially trained model ---")
            initial_ckpt = torch.load(best_initial_weights_path, map_location=device)
            model = initial_ckpt['model'].float()
            zero_top_weights(model, percentile=opt.prune_keep_percent)
            save_checkpoint(model, pruned_model_path)
        else:
            print(f"\n--- SKIPPING Step 2: Found existing pruned model at {pruned_model_path} ---")

        # Step 3: Average the pre-pruned and post-pruned models
        if not opt.resume_run or not averaged_model_path.exists():
            print("\n--- Step 3: Averaging pre-pruned and post-pruned weights ---")
            initial_ckpt = torch.load(best_initial_weights_path, map_location=device)
            pruned_ckpt = torch.load(pruned_model_path, map_location=device)
            
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
            callbacks = DriveSyncCallback(final_train_save_dir, drive_base_dir / final_train_name) if drive_base_dir else Callbacks()
            
            final_train_opt_dict = {
                'weights': str(averaged_model_path), 'cfg': opt.cfg, 'data': opt.data,
                'hyp': 'data/hyps/hyp.scratch-low.yaml', 'epochs': remaining_epochs,
                'batch_size': opt.batch_size, 'imgsz': opt.img_size,
                'project': opt.project, 'name': final_train_name, 'exist_ok': True,
                'device': str(device), 'cache': opt.cache, 'workers': opt.workers,
                'rect': True, 'resume': False, 'image_weights': False, 'nosave': False,
                'noval': False, 'noautoanchor': False, 'noplots': False, 'evolve': None,
                'bucket': '', 'multi_scale': False, 'single_cls': False, 'optimizer': 'SGD',
                'sync_bn': False, 'quad': False, 'cos_lr': False, 'label_smoothing': 0.0,
                'patience': 100, 'freeze': [0], 'save_period': -1, 'seed': 0, 'local_rank': -1,
                'entity': None, 'upload_dataset': False, 'bbox_interval': -1, 'artifact_alias': "latest"
            }
            # *** FIX: Call train() with the correct 4 arguments ***
            train(
                final_train_opt_dict['hyp'], 
                argparse.Namespace(**final_train_opt_dict), 
                device, 
                callbacks
            )
            print(f"Final training complete. Results saved to {final_train_save_dir}")
        else:
             print(f"\n--- SKIPPING Step 4 or No remaining epochs ---")

    finally:
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
    parser.add_argument('--project', default='runs/custom_train', help='local save directory')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    parser.add_argument('--pruning-epoch', type=int, default=10, help='Epochs for initial training BEFORE pruning.')
    parser.add_argument('--total-epochs', type=int, default=50, help='Total COMBINED epochs.')
    parser.add_argument('--prune-keep-percent', type=float, default=90.0, help='Percent of weights to KEEP after pruning.')
    parser.add_argument('--resume-run', action='store_true', help='Resume from the last successfully completed stage.')
    parser.add_argument('--save-to-drive', action='store_true', help='Save all results to Google Drive.')
    parser.add_argument('--drive-folder-path', type=str, default='YOLOv5_Runs', help='Base Google Drive folder')

    opt = parser.parse_args()

    if opt.total_epochs <= opt.pruning_epoch:
        raise ValueError("Error: --total-epochs must be greater than --pruning-epoch.")
        
    main(opt)
