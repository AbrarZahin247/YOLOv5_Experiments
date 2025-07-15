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
        # Use append mode 'a' to continue logging to the same file on resume
        self.logfile = open(filepath, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()  # Ensure logs are written immediately

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
    # MODIFIED: No longer need increment_path as we want predictable paths
    from utils.general import check_yaml
except ImportError as e:
    print("FATAL: Could not import YOLOv5 utilities. This script requires a YOLOv5 environment.")
    print(f"ImportError: {e}")
    print("Please run this script from the root of a cloned YOLOv5 repository.")
    sys.exit(1)

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
        url_v7 = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{weights}"
        download_success = download(url_v7, filename=str(model_path))
        if not download_success:
            print(f"Download from v7.0 failed. Trying v6.0 release...")
            url_v6 = f"https://github.com/ultralytics/yolov5/releases/download/v6.0/{weights}"
            download(url_v6, filename=str(model_path))

    if not model_path.exists():
        raise FileNotFoundError(f"'{weights}' does not exist and download failed.")
    # MODIFIED: Always set weights_only=False to avoid issues with YOLOv5's pickled objects
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = ckpt['model'].float()
    model.eval()
    if fuse and hasattr(model, 'fuse'):
        model.fuse()
    if device:
        model.to(device)
    if not inplace:
        model = model.copy()
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

def save_to_drive(src_path, drive_root_folder):
    if not drive_root_folder:
        print("Skipping save to drive: --save-to-drive not specified or path empty.")
        return

    drive_base = '/content/drive/My Drive'
    if not Path(drive_base).exists():
        print("Google Drive not mounted at /content/drive. Cannot save.")
        return

    try:
        src = Path(src_path)
        if not src.exists():
            print(f"Source path {src_path} does not exist. Cannot copy to Drive.")
            return

        dst = os.path.join(drive_base, drive_root_folder)
        os.makedirs(dst, exist_ok=True)

        if src.is_dir():
            dst_path = Path(dst) / src.name
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src, dst_path)
            print(f"Copied directory {src} to {dst_path}")
        elif src.is_file():
            shutil.copy2(src, dst)
            print(f"Copied file {src} to {dst}")
    except Exception as e:
        print(f"Could not save to drive: {e}")

def main(opt):
    # --- CONSTRUCT DESCRIPTIVE NAME FOR FOLDERS AND LOG FILE ---
    date_str = datetime.now().strftime('%Y-%m-%d')
    model_name = Path(opt.initial_weights).stem
    prune_percent = 100 - opt.prune_keep_percent
    prune_percent_str = f'{prune_percent:g}'

    descriptive_name = (
        f"{date_str}_{model_name}_pruned{prune_percent_str}pct_"
        f"retrain{opt.pruning_epoch}e_final{opt.total_epochs}e"
    )
    
    # --- SETUP PREDICTABLE PATHS FOR RESUMPTION ---
    os.makedirs(opt.project, exist_ok=True)
    log_file_path = os.path.join(opt.project, f"{descriptive_name}.log")
    pruned_weights_path = os.path.join(opt.project, f'{descriptive_name}_pruned.pt')
    averaged_weights_path = os.path.join(opt.project, f'{descriptive_name}_averaged.pt')
    
    # MODIFIED: Define explicit, non-incrementing run directories
    retrain_name = f'retrain_{descriptive_name}'
    final_train_name = f'final_{descriptive_name}'
    retrain_save_dir = Path(opt.project) / retrain_name
    final_save_dir = Path(opt.project) / final_train_name
    
    last_retrain_weights = retrain_save_dir / 'weights' / 'last.pt'
    best_retrain_weights = retrain_save_dir / 'weights' / 'best.pt'
    last_final_weights = final_save_dir / 'weights' / 'last.pt'
    best_final_weights = final_save_dir / 'weights' / 'best.pt'

    # --- SETUP LOGGING TO FILE ---
    original_stdout = sys.stdout
    sys.stdout = Logger(log_file_path) # Redirect stdout

    drive_root_path = None
    if opt.save_to_drive:
        drive_root_path = os.path.join(opt.drive_folder_path, descriptive_name)
    
    # Now that logging is active, print headers
    print(f"Starting run: {descriptive_name}")
    print(f"Run log file: {log_file_path}")
    print("="*60)
    if opt.save_to_drive:
        print(f"✅ All results will be saved to Google Drive under: '{drive_root_path}'")

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # PyTorch weights_only workaround
    _original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    try:
        # 1. Load and Prune Model
        if not opt.resume_run or not Path(pruned_weights_path).exists():
            print("\n--- Step 1 & 2: Loading and Pruning pre-trained model ---")
            model = attempt_load(opt.initial_weights, device=device)
            for param in model.parameters():
                param.requires_grad = True
            
            zero_top_weights(model, percentile=100 - opt.prune_keep_percent)
            
            # Save pruned model
            torch.save({'model': model}, pruned_weights_path)
            print(f"Pruned model weights saved to {pruned_weights_path}")
        else:
            print(f"\n--- SKIPPING Step 1 & 2: Found existing pruned model at {pruned_weights_path} ---")

        # 4. Retrain pruned model
        if not opt.resume_run or not best_retrain_weights.exists():
            print(f"\n--- Step 4: Retraining pruned model for {opt.pruning_epoch} epochs ---")
            retrain_opt = argparse.Namespace(
                weights=pruned_weights_path, cfg=opt.cfg, data=opt.data,
                hyp='data/hyps/hyp.scratch-low.yaml', epochs=opt.pruning_epoch,
                batch_size=opt.batch_size, imgsz=opt.img_size, rect=False,
                nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None,
                bucket='', cache=opt.cache, image_weights=False, device=str(device),
                multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False,
                workers=8, project=opt.project, name=retrain_name,
                exist_ok=True, # MODIFIED: Allow writing to existing directory
                quad=False, cos_lr=False, label_smoothing=0.0,
                patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1,
                entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias="latest"
            )
            # NEW: Check if we should resume this specific training step
            if opt.resume_run and last_retrain_weights.exists():
                print(f"Resuming retraining from checkpoint: {last_retrain_weights}")
                retrain_opt.resume = str(last_retrain_weights)
                retrain_opt.weights = '' # YOLOv5 expects weights to be empty when resuming
            else:
                retrain_opt.resume = False

            train(retrain_opt.hyp, retrain_opt, device, callbacks=Callbacks())
            print(f"Retraining complete. Results saved to {retrain_save_dir}")
            if opt.save_to_drive:
                save_to_drive(str(retrain_save_dir), drive_root_path)
        else:
            print(f"\n--- SKIPPING Step 4: Found completed retraining results in {retrain_save_dir} ---")

        # 5. Average pruned and retrained weights
        if not opt.resume_run or not Path(averaged_weights_path).exists():
            print("\n--- Step 5: Averaging pruned and retrained weights ---")
            pruned_model_ckpt = torch.load(pruned_weights_path, map_location=device)
            retrained_model_ckpt = torch.load(best_retrain_weights, map_location=device) # Use best weights
            averaged_model = pruned_model_ckpt['model']
            pruned_state_dict = averaged_model.state_dict()
            retrained_state_dict = retrained_model_ckpt['model'].state_dict()
            for key in pruned_state_dict:
                if key in retrained_state_dict:
                    p_tensor, r_tensor = pruned_state_dict[key], retrained_state_dict[key]
                    if p_tensor.shape == r_tensor.shape and p_tensor.dtype.is_floating_point:
                        pruned_state_dict[key].data = (p_tensor.data + r_tensor.data) / 2.0
            averaged_model.load_state_dict(pruned_state_dict)
            torch.save({'model': averaged_model}, averaged_weights_path)
            print(f"Averaged model weights saved to {averaged_weights_path}")
            if opt.save_to_drive:
                save_to_drive(averaged_weights_path, drive_root_path)
        else:
            print(f"\n--- SKIPPING Step 5: Found existing averaged model at {averaged_weights_path} ---")

        # 6. Final training
        if not opt.resume_run or not best_final_weights.exists():
            print(f"\n--- Step 6: Final training for {opt.total_epochs} epochs ---")
            final_train_opt = argparse.Namespace(
                weights=averaged_weights_path, cfg=opt.cfg, data=opt.data,
                hyp='data/hyps/hyp.scratch-low.yaml', epochs=opt.total_epochs,
                batch_size=opt.batch_size, imgsz=opt.img_size, rect=False,
                nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None,
                bucket='', cache=opt.cache, image_weights=False, device=str(device),
                multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False,
                workers=8, project=opt.project, name=final_train_name,
                exist_ok=True, # MODIFIED: Allow writing to existing directory
                quad=False, cos_lr=False, label_smoothing=0.0,
                patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1,
                entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias="latest"
            )
            # NEW: Check if we should resume this specific training step
            if opt.resume_run and last_final_weights.exists():
                print(f"Resuming final training from checkpoint: {last_final_weights}")
                final_train_opt.resume = str(last_final_weights)
                final_train_opt.weights = '' # YOLOv5 expects weights to be empty when resuming
            else:
                final_train_opt.resume = False

            train(final_train_opt.hyp, final_train_opt, device, callbacks=Callbacks())
            print(f"Final training complete. Results saved to {final_save_dir}")
            if opt.save_to_drive:
                save_to_drive(str(final_save_dir), drive_root_path)
        else:
             print(f"\n--- SKIPPING Step 6: Found completed final training results in {final_save_dir} ---")
             print("\nRun already completed. Nothing to do. ✨")

    finally:
        # --- Restore stdout and save the final log file ---
        if sys.stdout != original_stdout:
            sys.stdout.logfile.close() # Close the file handle
            sys.stdout = original_stdout # Restore original stdout
        
        print(f"\nTerminal output logging complete. Saved to {log_file_path}")
        if opt.save_to_drive and drive_root_path:
            save_to_drive(log_file_path, drive_root_path)

        # Restore other patched functions
        torch.load = _original_torch_load
        print("\nRestored original torch.load function.")
        print("\nProcess finished successfully. ✨")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 Pruning, Retraining, and Averaging with Full Logging and Resume Capability")
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--initial-weights', type=str, default='yolov5n.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--project', default='runs/custom_train', help='local save directory for logs and intermediate weights')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    parser.add_argument('--pruning-epoch', type=int, default=3, help='Epochs to retrain after pruning')
    parser.add_argument('--total-epochs', type=int, default=10, help='Epochs for final training')
    parser.add_argument('--prune-keep-percent', type=float, default=90.0, help='Percent of weights to KEEP (e.g., 90.0)')
    # NEW ARGUMENT
    parser.add_argument('--resume-run', action='store_true', help='Resume from the last saved state of the run')
    parser.add_argument('--save-to-drive', action='store_true', help='Save all results to a unique folder in Google Drive')
    parser.add_argument('--drive-folder-path', type=str, default='YOLOv5_Runs', help='Base Google Drive folder to save result folders into')

    opt = parser.parse_args()
    main(opt)
