import torch
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import sys
import shutil # Import shutil for save_to_drive

# ---- YOLOv5 IMPORT FIX ----
# This block is now more of a fallback, as changing the working directory is the primary fix.
if not (Path('train.py').exists() and Path('utils').is_dir()):
    yolov5_root = os.environ.get('YOLOV5_ROOT', '/content/yolov5')
    if Path(yolov5_root).exists() and str(yolov5_root) not in sys.path:
        sys.path.append(str(yolov5_root))

# Import YOLOv5 utilities
try:
    from train import train
    from utils.callbacks import Callbacks
    from utils.general import increment_path
except ImportError as e:
    print("FATAL: Could not import YOLOv5 utilities. This script requires a YOLOv5 environment.")
    print(f"ImportError: {e}")
    print("Please run this script from the root of a cloned YOLOv5 repository.")
    sys.exit(1) # Exit if imports fail

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

def save_to_drive(src_path, drive_folder_path):
    """Copies a file or directory to Google Drive."""
    if not drive_folder_path:
        print("Skipping save to drive: --drive-folder-path not specified.")
        return
    
    # This assumes you have mounted your drive at /content/drive
    drive_base = '/content/drive/My Drive'
    if not Path(drive_base).exists():
        print("Google Drive not mounted at /content/drive. Cannot save.")
        return
        
    dst = os.path.join(drive_base, drive_folder_path)
    try:
        src = Path(src_path)
        if not src.exists():
            print(f"Source path {src_path} does not exist. Cannot copy to Drive.")
            return

        if src.is_dir():
            dst_path = Path(dst) / src.name
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src, dst_path)
            print(f"Copied directory {src} to {dst_path}")
        elif src.is_file():
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied file {src} to {dst}")
    except Exception as e:
        print(f"Could not save to drive: {e}")


def main(opt):
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load pre-trained YOLOv5 model
    print("\n--- Step 1: Loading pre-trained model ---")
    model = attempt_load(opt.initial_weights, device=device)
    for param in model.parameters():
        param.requires_grad = True

    # 2. Zero out top X% of weights by magnitude (pruning)
    print(f"\n--- Step 2: Zeroing out top {100-opt.prune_keep_percent}% of weights ---")
    zero_top_weights(model, percentile=100 - opt.prune_keep_percent)

    # 3. Save pruned model
    pruned_weights_path = os.path.join(opt.project, 'yolov5_pruned.pt')
    os.makedirs(opt.project, exist_ok=True)
    torch.save({'model': model}, pruned_weights_path)
    print(f"Pruned model weights saved to {pruned_weights_path}")

    # PyTorch weights_only workaround
    _original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    
    try:
        # 4. Retrain pruned model
        print(f"\n--- Step 4: Retraining the pruned model for {opt.pruning_epoch} epochs ---")
        retrain_opt = argparse.Namespace(
            weights=pruned_weights_path, cfg=opt.cfg, data=opt.data,
            hyp='data/hyps/hyp.scratch-low.yaml', epochs=opt.pruning_epoch,
            batch_size=opt.batch_size, imgsz=opt.img_size, rect=False, resume=False,
            nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None,
            bucket='', cache=opt.cache, image_weights=False, device=str(device),
            multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False,
            workers=8, project='runs/train', name=f'exp_retrain_{opt.pruning_epoch}_epochs',
            exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0,
            patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1,
            entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias="latest"
        )
        # The train function returns the final results object which contains the save_dir
        results = train(retrain_opt.hyp, retrain_opt, device, callbacks=Callbacks())
        retrain_save_dir = results.save_dir
        print(f"Retraining complete. Results saved to {retrain_save_dir}")

        # LOGICAL FIX: Save to drive AFTER training is complete
        if opt.save_to_drive:
            save_to_drive(retrain_save_dir, opt.drive_folder_path)

        # 5. Average pruned and retrained weights
        print("\n--- Step 5: Averaging pruned and retrained weights ---")
        last_weights_path = Path(retrain_save_dir) / 'weights' / 'last.pt'
        pruned_model_ckpt = torch.load(pruned_weights_path, map_location=device)
        retrained_model_ckpt = torch.load(last_weights_path, map_location=device)

        averaged_model = pruned_model_ckpt['model'] # Start with the pruned model structure
        pruned_state_dict = averaged_model.state_dict()
        retrained_state_dict = retrained_model_ckpt['model'].state_dict()
        
        for key in pruned_state_dict:
            if key in retrained_state_dict:
                p_tensor = pruned_state_dict[key]
                r_tensor = retrained_state_dict[key]
                if p_tensor.shape == r_tensor.shape and p_tensor.dtype.is_floating_point:
                    pruned_state_dict[key].data = (p_tensor.data + r_tensor.data) / 2.0
        
        averaged_model.load_state_dict(pruned_state_dict)
        averaged_weights_path = os.path.join(opt.project, 'yolov5_averaged.pt')
        torch.save({'model': averaged_model}, averaged_weights_path)
        print(f"Averaged model weights saved to {averaged_weights_path}")
        if opt.save_to_drive:
            save_to_drive(averaged_weights_path, opt.drive_folder_path)

        # 6. Final training
        print(f"\n--- Step 6: Final training for {opt.total_epochs} more epochs ---")
        final_train_opt = argparse.Namespace(**vars(retrain_opt))
        final_train_opt.weights = averaged_weights_path
        final_train_opt.epochs = opt.total_epochs
        final_train_opt.name = f'exp_final_{opt.total_epochs}_epochs'
        
        final_results = train(final_train_opt.hyp, final_train_opt, device, callbacks=Callbacks())
        final_save_dir = final_results.save_dir
        print(f"Final training complete. Results saved to {final_save_dir}")

        # LOGICAL FIX: Save to drive AFTER final training is complete
        if opt.save_to_drive:
            save_to_drive(final_save_dir, opt.drive_folder_path)

    finally:
        torch.load = _original_torch_load
        print("\nRestored original torch.load function.")
        print("\nProcess finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--initial-weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--project', default='runs/custom_train', help='save directory')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')
    parser.add_argument('--pruning-epoch', type=int, default=3, help='Epochs to retrain after pruning')
    parser.add_argument('--total-epochs', type=int, default=5, help='Epochs for final training')
    parser.add_argument('--prune-keep-percent', type=float, default=90.0, help='Percent of weights to KEEP')
    parser.add_argument('--save-to-drive', action='store_true', help='Save results to Google Drive')
    parser.add_argument('--drive-folder-path', type=str, default='YOLOv5_Training_Results', help='Google Drive folder path')

    opt = parser.parse_args()
    main(opt)
