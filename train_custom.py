import torch
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import sys

# ---- YOLOv5 IMPORT FIX ----
# Automatically append YOLOv5 root to sys.path if not present
if not (Path('train.py').exists() and Path('utils').is_dir()):
    yolov5_root = os.environ.get('YOLOV5_ROOT', None)
    if yolov5_root and Path(yolov5_root).exists():
        sys.path.append(str(yolov5_root))
    else:
        sys.path.append('/content/yolov5')

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
        raise FileNotFoundError(f"'{weights}' does not exist and download failed from all sources.")
    # Set weights_only=False to load the full model object, not just weights
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = ckpt['model'].float()
    model.eval()
    if fuse and hasattr(model, 'fuse'):
        print("Fusing model...")
        model.fuse()
    if device:
        model.to(device)
    if not inplace:
        model = model.copy()
    logging.info(f'Model summary: {len(list(model.modules()))} modules, {sum(p.numel() for p in model.parameters())} parameters')
    return model

def get_model_weights_as_vector(model):
    weights_vector = []
    for param in model.parameters():
        if param.requires_grad:
            weights_vector.append(param.data.clone().cpu().numpy().flatten())
    if not weights_vector:
        raise ValueError("No parameters with requires_grad=True found in the model.")
    return np.concatenate(weights_vector)

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
    print(f"Zeroed out weights above the {percentile}th percentile (magnitude > {threshold:.4f})")

# Import YOLOv5 utilities
try:
    from train import train
    from utils.callbacks import Callbacks
    from utils.general import increment_path
except ImportError:
    print("Warning: Required YOLOv5 utilities could not be imported. The training steps will fail.")
    print("Please ensure this script is run within a YOLOv5 repository environment.")
    train = None
    Callbacks = None
    increment_path = None

def save_to_drive(src_path, drive_folder_path):
    """
    Copy a file or a directory to Google Drive.
    - If src_path is a directory, copies all contents (overwrites destination).
    - If src_path is a file, copies it to the destination directory.
    """
    import shutil
    import os

    dst = os.path.join('/content/drive/My Drive', drive_folder_path)
    try:
        if os.path.isdir(src_path):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src_path, dst)
            print(f"Copied directory {src_path} to {dst}")
        elif os.path.isfile(src_path):
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src_path, dst)
            print(f"Copied file {src_path} to {dst}")
        else:
            print(f"Source path {src_path} does not exist.")
    except Exception as e:
        print(f"Could not save to drive: {e}")

def main(opt):
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load pre-trained YOLOv5 model
    print("\n--- Step 1: Loading pre-trained model ---")
    model = attempt_load(opt.initial_weights, device=device)
    print("Model loaded successfully.")

    for param in model.parameters():
        param.requires_grad = True
    print("Set requires_grad=True for all model parameters.")

    # 2. Zero out top X% of weights by magnitude (pruning)
    print(f"\n--- Step 2: Zeroing out top {100-opt.prune_keep_percent}% of weights by magnitude ---")
    zero_top_weights(model, percentile=100 - opt.prune_keep_percent)

    # 3. Save pruned model
    pruned_weights_path = os.path.join(opt.project, 'yolov5_pruned.pt')
    os.makedirs(opt.project, exist_ok=True)
    save_dict = {'model': model, 'optimizer': None, 'epoch': -1}
    torch.save(save_dict, pruned_weights_path)
    print("Pruned model weights saved.")
    pruned_model_saved_folder_name = f"exp_retrain_{opt.pruning_epoch}_epochs"

    # Optionally save to Google Drive
    if opt.save_to_drive:
        base_drive_path = f'/content/yolov5/runs/train/{pruned_model_saved_folder_name}'
        save_to_drive(base_drive_path, opt.drive_folder_path)

    if not train or not Callbacks or not increment_path:
        print("\nSkipping training steps because YOLOv5 utilities are not available.")
        print("Process stopped after pruning and saving the model.")
        return

    # PyTorch weights_only workaround
    _original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    print("\nApplied temporary patch to torch.load for compatibility.")

    try:
        # 4. Retrain pruned model for opt.pruning_epoch epochs
        print(f"\n--- Step 4: Retraining the pruned model for {opt.pruning_epoch} epochs ---")
        hyp_path = 'data/hyps/hyp.scratch-low.yaml'
        cfg_path = opt.cfg or (model.yaml['cfg'] if hasattr(model, 'yaml') and 'cfg' in model.yaml else '')
        if not cfg_path:
            print("Warning: Model config (.yaml) not found. You may need to specify it with --cfg.")

        retrain_opt = argparse.Namespace(
            weights=pruned_weights_path,
            cfg=cfg_path,
            data=opt.data,
            hyp=hyp_path,
            epochs=opt.pruning_epoch,
            batch_size=opt.batch_size,
            imgsz=opt.img_size,
            rect=False,
            resume=False,
            nosave=False,
            noval=False,
            noautoanchor=False,
            noplots=False,
            evolve=None,
            bucket='',
            cache=opt.cache,
            image_weights=False,
            device=str(device),
            multi_scale=False,
            single_cls=False,
            adam=False,
            sync_bn=False,
            workers=8,
            project='runs/train',
            name=pruned_model_saved_folder_name,
            exist_ok=False,
            quad=False,
            cos_lr=False,
            label_smoothing=0.0,
            patience=100,
            freeze=[0],
            save_period=-1,
            local_rank=-1,
            entity=None,
            upload_dataset=False,
            bbox_interval=-1,
            artifact_alias="latest",
            seed=0,
            linear_lr=False,
            optimizer='SGD',
        )
        save_dir = Path(increment_path(Path(retrain_opt.project) / retrain_opt.name, exist_ok=retrain_opt.exist_ok))
        retrain_opt.save_dir = str(save_dir)

        callbacks = Callbacks()
        train(retrain_opt.hyp, retrain_opt, device, callbacks)
        print(f"Retraining complete after {opt.pruning_epoch} epochs.")

        # 5. Average pruned and retrained weights
        print("\n--- Step 5: Averaging pruned and retrained weights ---")
        last_weights_10_epochs = save_dir / 'weights' / 'last.pt'
        pruned_model_ckpt = torch.load(pruned_weights_path, map_location=device)
        retrained_model_ckpt = torch.load(last_weights_10_epochs, map_location=device)

        pruned_model = pruned_model_ckpt['model']
        retrained_model = retrained_model_ckpt['model']
        averaged_model = attempt_load(opt.initial_weights, device=device) # Load a fresh model to hold the result

        pruned_state_dict = pruned_model.state_dict()
        retrained_state_dict = retrained_model.state_dict()
        averaged_state_dict = averaged_model.state_dict()

        for key in averaged_state_dict:
            if key in pruned_state_dict and key in retrained_state_dict:
                p_tensor = pruned_state_dict[key]
                r_tensor = retrained_state_dict[key]
                
                # Check if tensor shapes are the same before averaging
                if p_tensor.shape == r_tensor.shape:
                    # If shapes match, average the weights
                    if p_tensor.dtype.is_floating_point:
                        averaged_state_dict[key].data = (p_tensor.data + r_tensor.data) / 2.0
                    else:
                        averaged_state_dict[key].data = p_tensor.data # Keep non-float data
                else:
                    # If shapes mismatch (e.g., final detection layer), use the retrained weights
                    print(f"Shape mismatch for key '{key}': pruned {p_tensor.shape} vs retrained {r_tensor.shape}. Using retrained weights.")
                    averaged_state_dict[key].data = r_tensor.data
            elif key in retrained_state_dict:
                # Fallback to retrained weights if key is not in pruned model for some reason
                averaged_state_dict[key].data = retrained_state_dict[key].data
        
        averaged_model.load_state_dict(averaged_state_dict)
        averaged_weights_path = os.path.join(opt.project, 'yolov5_averaged.pt')
        avg_save_dict = {'model': averaged_model, 'optimizer': None, 'epoch': -1}
        torch.save(avg_save_dict, averaged_weights_path)
        print(f"Averaged model weights saved to {averaged_weights_path}")


        if opt.save_to_drive:
            save_to_drive(averaged_weights_path, opt.drive_folder_path)

        # 6. Final training for opt.total_epochs epochs
        print(f"\n--- Step 6: Continuing training with averaged weights for {opt.total_epochs} more epochs ---")
        final_train_opt = argparse.Namespace(**vars(retrain_opt))
        final_train_opt.weights = averaged_weights_path
        final_train_opt.epochs = opt.total_epochs
        final_train_opt.name = f'exp_final_{opt.total_epochs}_epochs'
        final_train_opt.seed = 0
        final_save_dir = Path(increment_path(Path(final_train_opt.project) / final_train_opt.name, exist_ok=final_train_opt.exist_ok))
        final_train_opt.save_dir = str(final_save_dir)

        train(final_train_opt.hyp, final_train_opt, device, callbacks)
        print(f"Final training complete.")

        if opt.save_to_drive:
            base_drive_path = f'/content/yolov5/runs/train/{final_train_opt.name}'
            save_to_drive(base_drive_path, opt.drive_folder_path)

    finally:
        torch.load = _original_torch_load
        print("\nRestored original torch.load function.")
        print("\nProcess finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 Pruning, Retraining, and Averaging Script")
    parser.add_argument('--data', type=str, default='data/signature_data.yaml', help='path to your data.yaml file')
    parser.add_argument('--initial-weights', type=str, default='yolov5s.pt', help='initial weights path (e.g., yolov5s.pt, yolov5m.pt)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path (optional)')
    parser.add_argument('--img-size', type=int, default=640, help='image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    parser.add_argument('--project', default='weights', help='save directory for pruned/averaged weights')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    parser.add_argument('--pruning-epoch', type=int, default=10, help='Epochs to retrain after pruning')
    parser.add_argument('--total-epochs', type=int, default=20, help='Epochs to continue after averaging')
    parser.add_argument('--prune-keep-percent', type=float, default=90, help='Percent of weights to keep (top weights are zeroed out)')
    parser.add_argument('--save-to-drive', action='store_true', help='Save results to Google Drive')
    parser.add_argument('--drive-folder-path', type=str, default='', help='Google Drive folder to save results')

    # Use parse_known_args to avoid conflicts if running in environments like Jupyter
    opt, _ = parser.parse_known_args()
    print("--- Script Start ---")
    main(opt)
