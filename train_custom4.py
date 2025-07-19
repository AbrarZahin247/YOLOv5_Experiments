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
if not (Path('train.py').exists() and Path('utils').is_dir()):
    yolov5_root = os.environ.get('YOLOV5_ROOT', '/content/yolov5')
    if Path(yolov5_root).exists() and str(yolov5_root) not in sys.path:
        sys.path.append(str(yolov5_root))

# Import YOLOv5 utilities
try:
    from train import train
    from utils.callbacks import Callbacks
    from utils.general import check_yaml
except ImportError as e:
    print("FATAL: Could not import YOLOv5 utilities. This script requires a YOLOv5 environment.")
    print(f"ImportError: {e}")
    sys.exit(1)

# ---- OPTIMIZED DYNAMIC DRIVE SAVING CALLBACK ----
class DriveSyncCallback(Callbacks):
    """
    An optimized YOLOv5 callback to save files to Google Drive without blocking training.
    - Saves best.pt only when a new best model is found.
    - Saves last.pt and results.csv every `save_period` epochs.
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
        # Save best.pt only when it's updated
        if best:
            local_best_file = self.weights_dir / 'best.pt'
            if local_best_file.exists():
                try:
                    shutil.copy2(str(local_best_file), self.drive_weights_dir)
                except Exception as e:
                    print(f"WARNING: Failed to copy best.pt to Drive: {e}")

        # Save last.pt and results.csv periodically
        if epoch > 0 and epoch % self.save_period == 0:
            local_last_file = self.weights_dir / 'last.pt'
            if local_last_file.exists():
                try:
                    shutil.copy2(str(local_last_file), self.drive_weights_dir)
                except Exception as e:
                    print(f"WARNING: Failed to copy last.pt to Drive: {e}")
            
            results_csv = self.local_run_dir / 'results.csv'
            if results_csv.exists():
                try:
                    shutil.copy2(str(results_csv), self.drive_run_dir)
                except Exception as e:
                    print(f"WARNING: Failed to copy results.csv to Drive: {e}")

def download(url, filename="yolov5s.pt"):
    print(f"Downloading {filename} from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                total_size = int(r.headers.get('content-length', 0))
                pbar = tqdm(total=total_size, desc=f"Downloading {filename}", unit='iB', unit_scale=True)
                for chunk in r.iter_content(chunk_size=8192):
                    pbar.update(len(chunk))
                    f.write(chunk)
                pbar.close()
        print(f"\nDownload completed successfully. Saved as {filename}")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False

def attempt_load(weights, device=None, inplace=True, fuse=True):
    model_path = Path(weights)
    if not model_path.exists():
        print(f"'{weights}' not found. Attempting to download...")
        url = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{weights}"
        download(url, filename=str(model_path))

    if not model_path.exists():
        raise FileNotFoundError(f"'{weights}' does not exist and download failed.")
    # `weights_only=False` is necessary for your custom saving format {'model': model}
    ckpt = torch.load(weights, map_location=device, weights_only=False) 
    model = ckpt['model'].float()
    model.eval()
    if fuse and hasattr(model, 'fuse'):
        model.fuse()
    if device:
        model.to(device)
    return model

def get_model_weights_as_vector(model):
    return np.concatenate([p.data.clone().cpu().numpy().flatten() for p in model.parameters() if p.requires_grad])

def set_model_weights_from_vector(model, weights_vector):
    pointer = 0
    for param in model.parameters():
        if param.requires_grad:
            num_param = param.numel()
            param.data = torch.from_numpy(weights_vector[pointer:pointer + num_param]).view_as(param).to(param.device)
            pointer += num_param

def zero_top_weights(model, percentile=90):
    weights = get_model_weights_as_vector(model)
    threshold = np.percentile(np.abs(weights), percentile)
    weights[np.abs(weights) > threshold] = 0
    set_model_weights_from_vector(model, weights)
    print(f"Zeroed weights above the {percentile}th percentile (magnitude > {threshold:.4f})")

def save_to_drive(src_path, drive_dest_path):
    """A simplified function to save a single file to a drive location."""
    if not drive_dest_path:
        return
    try:
        os.makedirs(os.path.dirname(drive_dest_path), exist_ok=True)
        shutil.copy2(src_path, drive_dest_path)
        print(f"Saved {src_path} to {drive_dest_path}")
    except Exception as e:
        print(f"Could not save {src_path} to drive: {e}")

# ##################################################################
# ################## MAIN FUNCTION (LOGIC MODIFIED) ################
# ##################################################################
def main(opt):
    # --- SETUP DESCRIPTIVE NAMES AND PATHS ---
    date_str = datetime.now().strftime('%Y-%m-%d')
    model_name = Path(opt.initial_weights).stem
    prune_percent = 100 - opt.prune_keep_percent
    prune_percent_str = f'{prune_percent:g}'

    # Updated descriptive name to reflect the new workflow
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
    
    # --- SETUP DRIVE PATHS ---
    drive_base_dir = None
    if opt.save_to_drive:
        drive_base_dir = Path('/content/drive/My Drive') / opt.drive_folder_path / descriptive_name
        print(f"Google Drive sync is enabled. Base path: {drive_base_dir}")

    # --- SETUP LOGGING ---
    original_stdout = sys.stdout
    sys.stdout = Logger(str(log_file_path))
    
    print(f"Starting run: {descriptive_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Temporarily patch torch.load for your custom model checkpoints
    _original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    try:
        # Step 1: Initial Training
        if not opt.resume_run or not (initial_train_save_dir / 'weights' / 'best.pt').exists():
            print(f"\n--- Step 1: Initial training for {opt.pruning_epoch} epochs ---")
            callbacks = Callbacks()
            if drive_base_dir:
                drive_initial_dir = drive_base_dir / initial_train_save_dir.name
                callbacks = DriveSyncCallback(local_run_dir=initial_train_save_dir, drive_run_dir=drive_initial_dir, save_period=10)
            
            initial_train_opt = argparse.Namespace(
                weights=opt.initial_weights, cfg=opt.cfg, data=opt.data,
                hyp='data/hyps/hyp.scratch-low.yaml', epochs=opt.pruning_epoch,
                batch_size=opt.batch_size, imgsz=opt.img_size,
                project=opt.project, name=initial_train_name, exist_ok=True,
                device=str(device), cache=opt.cache, workers=opt.workers, 
                rect=True, resume=False, image_weights=False,
                nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket='',
                multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False, quad=False,
                cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1,
                seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1,
                artifact_alias="latest"
            )
            initial_train_opt.save_dir = str(initial_train_save_dir)
            if opt.resume_run and (initial_train_save_dir / 'weights' / 'last.pt').exists():
                initial_train_opt.resume = True
                initial_train_opt.weights = ''
            
            train(initial_train_opt.hyp, initial_train_opt, device, callbacks)
            print(f"Initial training complete. Results saved to {initial_train_save_dir}")
        else:
            print(f"\n--- SKIPPING Step 1: Found completed initial training results in {initial_train_save_dir} ---")

        # Step 2: Prune the initially trained model
        if not opt.resume_run or not pruned_model_path.exists():
            print("\n--- Step 2: Pruning the initially trained model ---")
            best_initial_weights = initial_train_save_dir / 'weights' / 'best.pt'
            model = attempt_load(str(best_initial_weights), device=device)
            for param in model.parameters():
                param.requires_grad = True
            zero_top_weights(model, percentile=100 - opt.prune_keep_percent)
            torch.save({'model': model}, pruned_model_path)
            print(f"Pruned model weights saved to {pruned_model_path}")
            if drive_base_dir:
                save_to_drive(str(pruned_model_path), drive_base_dir / pruned_model_path.name)
        else:
            print(f"\n--- SKIPPING Step 2: Found existing pruned model at {pruned_model_path} ---")

        # Step 3: Average the pre-pruned and post-pruned models
        if not opt.resume_run or not averaged_model_path.exists():
            print("\n--- Step 3: Averaging pre-pruned and post-pruned weights ---")
            best_initial_weights = initial_train_save_dir / 'weights' / 'best.pt'
            initial_model_ckpt = torch.load(str(best_initial_weights), map_location=device)
            pruned_model_ckpt = torch.load(pruned_model_path, map_location=device)
            
            averaged_model = initial_model_ckpt['model'] # Start with the initial model structure
            initial_state_dict = averaged_model.state_dict()
            pruned_state_dict = pruned_model_ckpt['model'].state_dict()

            for key in initial_state_dict:
                if key in pruned_state_dict and initial_state_dict[key].dtype.is_floating_point:
                    initial_state_dict[key].data = (initial_state_dict[key].data + pruned_state_dict[key].data) / 2.0
            
            averaged_model.load_state_dict(initial_state_dict)
            torch.save({'model': averaged_model}, averaged_model_path)
            print(f"Averaged model weights saved to {averaged_model_path}")
            if drive_base_dir:
                save_to_drive(str(averaged_model_path), drive_base_dir / averaged_model_path.name)
        else:
            print(f"\n--- SKIPPING Step 3: Found existing averaged model at {averaged_model_path} ---")

        # Step 4: Final Training
        remaining_epochs = opt.total_epochs - opt.pruning_epoch
        if remaining_epochs <= 0:
            print(f"\nProcess finished after averaging. No remaining epochs for final training.")
        elif not opt.resume_run or not (final_train_save_dir / 'weights' / 'best.pt').exists():
            print(f"\n--- Step 4: Final training for remaining {remaining_epochs} epochs ---")
            callbacks = Callbacks()
            if drive_base_dir:
                drive_final_dir = drive_base_dir / final_train_save_dir.name
                callbacks = DriveSyncCallback(local_run_dir=final_train_save_dir, drive_run_dir=drive_final_dir, save_period=10)
            
            final_train_opt = argparse.Namespace(
                weights=str(averaged_model_path), cfg=opt.cfg, data=opt.data,
                hyp='data/hyps/hyp.scratch-low.yaml', epochs=remaining_epochs,
                batch_size=opt.batch_size, imgsz=opt.img_size,
                project=opt.project, name=final_train_name, exist_ok=True,
                device=str(device), cache=opt.cache, workers=opt.workers,
                rect=True, resume=False, image_weights=False,
                nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket='',
                multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False, quad=False,
                cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1,
                seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1,
                artifact_alias="latest"
            )
            final_train_opt.save_dir = str(final_train_save_dir)
            if opt.resume_run and (final_train_save_dir / 'weights' / 'last.pt').exists():
                final_train_opt.resume = True
                final_train_opt.weights = ''

            train(final_train_opt.hyp, final_train_opt, device, callbacks)
            print(f"Final training complete. Results saved to {final_train_save_dir}")
        else:
             print(f"\n--- SKIPPING Step 4: Found completed final training results in {final_train_save_dir} ---")

    finally:
        # --- Final cleanup and sync ---
        if sys.stdout != original_stdout:
            sys.stdout.logfile.close()
            sys.stdout = original_stdout
        
        print(f"\nTerminal output logging complete. Saved to {log_file_path}")
        
        if drive_base_dir:
            print("\n--- Performing final sync to Google Drive ---")
            save_to_drive(str(log_file_path), drive_base_dir / log_file_path.name)
            for run_dir in [initial_train_save_dir, final_train_save_dir]:
                final_best_pt = run_dir / 'weights' / 'best.pt'
                if final_best_pt.exists():
                     save_to_drive(str(final_best_pt), drive_base_dir / run_dir.name / 'weights' / 'best.pt')
                final_results_csv = run_dir / 'results.csv'
                if final_results_csv.exists():
                    save_to_drive(str(final_results_csv), drive_base_dir / run_dir.name / 'results.csv')

        torch.load = _original_torch_load
        print("\nRestored original torch.load function.")
        print("\nProcess finished successfully. ✨")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 Multi-Stage Training and Pruning with Dynamic Drive Sync")
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--initial-weights', type=str, default='yolov5n.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (increase for better GPU utilization)')
    parser.add_argument('--project', default='runs/custom_train', help='local save directory')
    parser.add_argument('--workers', type=int, default=2, help='max dataloader workers (2 is good for Colab)')
    parser.add_argument('--cache', action='store_true', default=True, help='cache images for faster training')
    
    # --- MODIFIED ARGUMENT HELP TEXTS ---
    parser.add_argument('--pruning-epoch', type=int, default=10, help='Epochs for initial training BEFORE pruning is applied.')
    parser.add_argument('--total-epochs', type=int, default=50, help='Total COMBINED epochs for both initial and final training phases.')
    
    parser.add_argument('--prune-keep-percent', type=float, default=90.0, help='Percent of weights to KEEP after pruning.')
    parser.add_argument('--resume-run', action='store_true', help='Resume from the last successfully completed stage.')
    parser.add_argument('--save-to-drive', action='store_true', help='Save all results to Google Drive.')
    parser.add_argument('--drive-folder-path', type=str, default='YOLOv5_Runs', help='Base Google Drive folder')

    opt = parser.parse_args()

    if opt.total_epochs <= opt.pruning_epoch:
        raise ValueError("Error: --total-epochs must be greater than --pruning-epoch.")
        
    main(opt)
