import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
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

# ---- UPDATED AND ROBUST Drive Sync Callback ----
class DriveSyncCallback(Callbacks):
    """
    A YOLOv5 callback to sync the ENTIRE project directory to Google Drive.
    """
    def __init__(self, local_project_dir, drive_project_dir):
        super().__init__()
        self.local_project_dir = Path(local_project_dir)
        self.drive_project_dir = Path(drive_project_dir)
        self.drive_project_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Drive Sync enabled. Project '{self.local_project_dir}' will be synced to '{self.drive_project_dir}'.")

    def sync_now(self, stage_name=""):
        """Performs an immediate, full sync of the project directory."""
        if not self.local_project_dir.is_dir():
            print(f"WARNING: Source project directory '{self.local_project_dir}' not found. Skipping sync.")
            return
            
        header = f"--- Immediate Sync after {stage_name} ---" if stage_name else "--- Performing Immediate Sync ---"
        print(f"\n{header}")
        try:
            # For a clean sync, first remove the old destination directory if it exists.
            if self.drive_project_dir.exists():
                shutil.rmtree(str(self.drive_project_dir))
            
            # Copy the entire local project directory to the destination.
            shutil.copytree(str(self.local_project_dir), str(self.drive_project_dir))
            print(f"Successfully synced all project files to '{self.drive_project_dir}'.")
        except Exception as e:
            print(f"WARNING: An error occurred during the sync to Drive: {e}")

    def on_fit_epoch_end(self, epoch, *args, **kwargs):
        """Runs at the end of each epoch to sync the entire project folder."""
        print(f"\n--- Epoch {epoch} complete. Syncing project to Drive. ---")
        self.sync_now(stage_name=f"Epoch {epoch}")


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
    
    initial_train_name = 'initial_train'
    final_train_name = 'final_train'
    initial_train_save_dir = local_project_dir / initial_train_name
    final_train_save_dir = local_project_dir / final_train_name
    
    drive_base_dir = None
    if opt.save_to_drive:
        drive_base_dir = Path('/content/drive/My Drive') / opt.drive_folder_path / descriptive_name

    original_stdout = sys.stdout
    sys.stdout = Logger(str(log_file_path))
    
    print(f"Starting run: {descriptive_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- SETUP CALLBACKS ---
    callbacks = Callbacks()
    if opt.save_to_drive and drive_base_dir:
        callbacks = DriveSyncCallback(local_project_dir, drive_base_dir)

    try:
        # Step 1: Initial Training
        if not opt.resume_run or not (initial_train_save_dir / 'weights' / 'best.pt').exists():
            print(f"\n--- Step 1: Initial training for {opt.pruning_epoch} epochs ---")
            train_opt_ns = argparse.Namespace(**vars(opt)) # Propagate all cmd args
            train_opt_ns.project = str(local_project_dir)
            train_opt_ns.name = initial_train_name
            train_opt_ns.exist_ok = True
            train_opt_ns.epochs = opt.pruning_epoch
            train_opt_ns.weights = opt.initial_weights
            train_opt_ns.hyp = 'data/hyps/hyp.scratch-low.yaml'
            train_opt_ns.save_dir = str(initial_train_save_dir)
            train(train_opt_ns.hyp, train_opt_ns, device, callbacks)
        else:
            print(f"\n--- SKIPPING Step 1: Found completed initial training results ---")

        # Step 2: Prune the initially trained model
        best_initial_weights_path = initial_train_save_dir / 'weights' / 'best.pt'
        if not opt.resume_run or not pruned_model_path.exists():
            print("\n--- Step 2: Pruning the initially trained model ---")
            initial_ckpt = torch.load(best_initial_weights_path, map_location=device, weights_only=False)
            model = initial_ckpt['model'].float()
            for param in model.parameters():
                param.requires_grad = True
            zero_top_weights(model, percentile=opt.prune_keep_percent)
            save_checkpoint(model, pruned_model_path)
            # Immediately sync after creating the file
            if opt.save_to_drive: callbacks.sync_now(stage_name="Pruning")
        else:
            print(f"\n--- SKIPPING Step 2: Found existing pruned model ---")

        # Step 3: Average the pre-pruned and post-pruned models
        if not opt.resume_run or not averaged_model_path.exists():
            print("\n--- Step 3: Averaging pre-pruned and post-pruned weights ---")
            initial_ckpt = torch.load(best_initial_weights_path, map_location=device, weights_only=False)
            pruned_ckpt = torch.load(pruned_model_path, map_location=device, weights_only=False)
            avg_model = initial_ckpt['model'].float()
            initial_state_dict = avg_model.state_dict()
            pruned_state_dict = pruned_ckpt['model'].state_dict()
            for key in initial_state_dict:
                if key in pruned_state_dict and initial_state_dict[key].dtype.is_floating_point:
                    initial_state_dict[key].data = (initial_state_dict[key].data + pruned_state_dict[key].data) / 2.0
            avg_model.load_state_dict(initial_state_dict)
            save_checkpoint(avg_model, averaged_model_path)
            # Immediately sync after creating the file
            if opt.save_to_drive: callbacks.sync_now(stage_name="Averaging")
        else:
            print(f"\n--- SKIPPING Step 3: Found existing averaged model ---")

        # Step 4: Final Training
        remaining_epochs = opt.total_epochs - opt.pruning_epoch
        if remaining_epochs > 0 and (not opt.resume_run or not (final_train_save_dir / 'weights' / 'best.pt').exists()):
            print(f"\n--- Step 4: Final training for remaining {remaining_epochs} epochs ---")
            train_opt_ns = argparse.Namespace(**vars(opt)) # Propagate all cmd args
            train_opt_ns.project = str(local_project_dir)
            train_opt_ns.name = final_train_name
            train_opt_ns.exist_ok = True
            train_opt_ns.epochs = remaining_epochs
            train_opt_ns.weights = str(averaged_model_path) # Start from the averaged model
            train_opt_ns.hyp = 'data/hyps/hyp.scratch-low.yaml'
            train_opt_ns.save_dir = str(final_train_save_dir)
            train(train_opt_ns.hyp, train_opt_ns, device, callbacks)
        else:
             print(f"\n--- SKIPPING Step 4 or No remaining epochs ---")

    finally:
        # Restore standard output and print final message
        if 'original_stdout' in locals() and sys.stdout != original_stdout:
            sys.stdout.logfile.close()
            sys.stdout = original_stdout
        
        print(f"\nTerminal output logging complete. Saved to {log_file_path}")
        print("\nProcess finished successfully. ✨")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 Multi-Stage Training and Pruning")
    
    # --- Key Training Arguments ---
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--initial-weights', type=str, default='yolov5n.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    
    # --- Custom Script Arguments ---
    parser.add_argument('--project', default='runs/custom_train', help='Base local directory for all runs')
    parser.add_argument('--pruning-epoch', type=int, default=10, help='Epochs for initial training BEFORE pruning.')
    parser.add_argument('--total-epochs', type=int, default=50, help='Total COMBINED epochs.')
    parser.add_argument('--prune-keep-percent', type=float, default=90.0, help='Percent of weights to KEEP after pruning.')
    parser.add_argument('--resume-run', action='store_true', help='Resume from the last successfully completed stage.')
    
    # --- Drive Sync Arguments ---
    parser.add_argument('--save-to-drive', action='store_true', help='Save all results to Google Drive.')
    parser.add_argument('--drive-folder-path', type=str, default='YOLOv5_Runs', help='Base Google Drive folder for all runs')

    opt = parser.parse_args()

    if opt.total_epochs <= opt.pruning_epoch:
        raise ValueError("Error: --total-epochs must be greater than --pruning-epoch.")
        
    main(opt)
