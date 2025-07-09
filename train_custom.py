import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm

# This script is designed to be run from the root of a YOLOv5 repository.
# It assumes that the 'train.py' script and the required data configurations are available.

def download(url, filename="yolov5s.pt"):
    """
    Downloads a file from a given URL with a progress bar.

    Args:
        url (str): The URL to download the file from.
        filename (str): The name to save the file as.

    Returns:
        bool: True if download is successful, False otherwise.
    """
    print(f"Downloading {filename} from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            # Check if the request was successful
            r.raise_for_status()
            # Open the file in binary write mode
            with open(filename, 'wb') as f:
                # Get the total file size from the headers
                total_size = int(r.headers.get('content-length', 0))
                # Create a progress bar with tqdm
                pbar = tqdm(total=total_size, desc=f"Downloading {filename}", unit='iB', unit_scale=True)
                # Iterate over the content in chunks
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
    """
    Loads a YOLOv5 model from a checkpoint file.
    If the file doesn't exist, it attempts to download it.

    Args:
        weights (str or Path): Path to the model weights file (.pt).
        device (torch.device, optional): The device to load the model onto.
        inplace (bool, optional): Whether to modify the model in-place.
        fuse (bool, optional): Whether to fuse Conv2d + BatchNorm2d layers.

    Returns:
        (torch.nn.Module): The loaded PyTorch model.
    """
    model_path = Path(weights)
    if not model_path.exists():
        print(f"'{weights}' not found. Attempting to download...")
        # Attempt to download from the v7.0 release
        url_v7 = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{weights}"
        download_success = download(url_v7, filename=str(model_path))
        
        # If v7.0 fails, fallback to the v6.0 release
        if not download_success:
             print(f"Download from v7.0 failed. Trying v6.0 release...")
             url_v6 = f"https://github.com/ultralytics/yolov5/releases/download/v6.0/{weights}"
             download(url_v6, filename=str(model_path))

    # After attempting to download, check if the file exists
    if not model_path.exists():
        raise FileNotFoundError(f"'{weights}' does not exist and download failed from all sources.")

    # --- FIX for PyTorch 2.6 ---
    # The `weights_only` argument is set to `False` because YOLOv5 .pt files
    # contain not just weights but also model architecture and metadata, which
    # would cause an error with the new default setting of `weights_only=True`.
    # This is safe as we are downloading from the official, trusted repository.
    ckpt = torch.load(weights, map_location=device, weights_only=False)

    # The checkpoint dictionary contains the model architecture itself
    model = ckpt['model'].float() 

    # Set the model to evaluation mode
    model.eval()

    # Fuse Conv2d and BatchNorm2d layers for faster inference
    if fuse and hasattr(model, 'fuse'):
        print("Fusing model...")
        model.fuse()

    # Move model to the specified device
    if device:
        model.to(device)

    # If not inplace, return a copy of the model
    if not inplace:
        model = model.copy()

    # Log a summary of the loaded model
    logging.info(f'Model summary: {len(list(model.modules()))} modules, {sum(p.numel() for p in model.parameters())} parameters')

    return model

def get_model_weights_as_vector(model):
    """Extracts and flattens all model weights into a single vector."""
    weights_vector = []
    for param in model.parameters():
        if param.requires_grad:
            weights_vector.append(param.data.clone().cpu().numpy().flatten())
    
    if not weights_vector:
        # This will now be prevented by setting requires_grad=True in main
        raise ValueError("No parameters with requires_grad=True found in the model.")
        
    return np.concatenate(weights_vector)

def set_model_weights_from_vector(model, weights_vector):
    """Sets model weights from a flattened vector."""
    pointer = 0
    for param in model.parameters():
        if param.requires_grad:
            num_param = param.numel()
            param.data = torch.from_numpy(
                weights_vector[pointer : pointer + num_param]
            ).view_as(param).to(param.device)
            pointer += num_param

def zero_top_weights(model, percentile=90):
    """
    Zeros out the top percentage of weights in a model based on magnitude.
    """
    weights = get_model_weights_as_vector(model)
    threshold = np.percentile(np.abs(weights), percentile)
    weights[np.abs(weights) > threshold] = 0
    set_model_weights_from_vector(model, weights)
    print(f"Zeroed out weights above the {percentile}th percentile (magnitude > {threshold:.4f})")

# Import the train function. This will fail if not in a YOLOv5 environment.
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

def main(opt):
    """Main function to run the YOLOv5 pruning and retraining process."""
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Step 1: Load a pre-trained YOLOv5 model ---
    print("\n--- Step 1: Loading pre-trained model ---")
    model = attempt_load(opt.initial_weights, device=device)
    print("Model loaded successfully.")

    # --- FIX for ValueError ---
    # The loaded model's parameters might have requires_grad=False. 
    # We set this to True to enable gradient calculations for pruning and retraining.
    for param in model.parameters():
        param.requires_grad = True
    print("Set requires_grad=True for all model parameters.")

    # --- Step 2: Zero out the 10% of weights with the highest magnitude ---
    print("\n--- Step 2: Zeroing out top 10% of weights by magnitude ---")
    zero_top_weights(model, percentile=90)

    # --- Step 3: Save the pruned model weights ---
    pruned_weights_path = os.path.join(opt.project, 'yolov5s_pruned.pt')
    print(f"\n--- Step 3: Saving pruned model weights to {pruned_weights_path} ---")
    os.makedirs(opt.project, exist_ok=True)
    # Save the model state in a dictionary compatible with YOLOv5's training script
    save_dict = {'model': model, 'optimizer': None, 'epoch': -1}
    torch.save(save_dict, pruned_weights_path)
    print("Pruned model weights saved.")

    # Exit if the train function is not available
    if not train or not Callbacks or not increment_path:
        print("\nSkipping training steps because YOLOv5 utilities are not available.")
        print("Process stopped after pruning and saving the model.")
        return

    # --- FIX for PyTorch 2.6 UnpicklingError in train.py ---
    # The train.py script from YOLOv5 loads checkpoints using torch.load(), which
    # in PyTorch 2.6 defaults to weights_only=True. This causes an error when
    # loading checkpoints that contain model objects. Since we cannot edit train.py,
    # we monkey-patch torch.load to force weights_only=False for the duration
    # of the training calls. This is a targeted workaround.
    _original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    print("\nApplied temporary patch to torch.load for compatibility with PyTorch 2.6+.")

    try:
        # --- Step 4: Retrain the pruned model for 10 epochs ---
        print("\n--- Step 4: Retraining the pruned model for 10 epochs ---")
        hyp_path = 'data/hyps/hyp.scratch-low.yaml'
        # Ensure a config file path is available
        cfg_path = opt.cfg or (model.yaml['cfg'] if hasattr(model, 'yaml') and 'cfg' in model.yaml else '')
        if not cfg_path:
            print("Warning: Model config (.yaml) not found. You may need to specify it with --cfg.")

        retrain_opt = argparse.Namespace(
            weights=pruned_weights_path,
            cfg=cfg_path,
            data=opt.data,
            hyp=hyp_path,
            epochs=10,
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
            name='exp_retrain_10_epochs',
            exist_ok=False,
            quad=False,
            linear_lr=False,
            label_smoothing=0.0,
            patience=100,
            freeze=[0], 
            save_period=-1,
            local_rank=-1,
            entity=None,
            upload_dataset=False,
            bbox_interval=-1,
            artifact_alias="latest",
            save_initial_weights=False,
            rewind_weights=False,
            optimizer='SGD',
            cos_lr=False # --- FIX for AttributeError ---
        )
        
        # Manually create and set the `save_dir` attribute
        save_dir = Path(increment_path(Path(retrain_opt.project) / retrain_opt.name, exist_ok=retrain_opt.exist_ok))
        retrain_opt.save_dir = str(save_dir)

        callbacks = Callbacks()
        train(retrain_opt.hyp, retrain_opt, device, callbacks)
        print(f"Retraining complete after 10 epochs.")

        # --- Step 5: Average the pruned and retrained weights ---
        print("\n--- Step 5: Averaging pruned and retrained weights ---")
        
        last_weights_10_epochs = save_dir / 'weights' / 'last.pt'

        pruned_model_ckpt = torch.load(pruned_weights_path, map_location=device) # Already patched
        retrained_model_ckpt = torch.load(last_weights_10_epochs, map_location=device) # Already patched

        pruned_model = pruned_model_ckpt['model']
        retrained_model = retrained_model_ckpt['model']

        averaged_model = attempt_load(opt.initial_weights, device=device)
        
        pruned_state_dict = pruned_model.state_dict()
        retrained_state_dict = retrained_model.state_dict()
        averaged_state_dict = averaged_model.state_dict()

        for key in pruned_state_dict:
            if key in retrained_state_dict and pruned_state_dict[key].data.dtype.is_floating_point:
                averaged_state_dict[key].data = (pruned_state_dict[key].data + retrained_state_dict[key].data) / 2.0
            else:
                averaged_state_dict[key].data = pruned_state_dict[key].data
                
        averaged_model.load_state_dict(averaged_state_dict)

        averaged_weights_path = os.path.join(opt.project, 'yolov5s_averaged.pt')
        avg_save_dict = {'model': averaged_model, 'optimizer': None, 'epoch': -1}
        torch.save(avg_save_dict, averaged_weights_path)
        print(f"Averaged model weights saved to {averaged_weights_path}")

        # --- Step 6: Continue training with averaged weights for 20 epochs ---
        print("\n--- Step 6: Continuing training with averaged weights for 20 more epochs ---")
        final_train_opt = retrain_opt
        final_train_opt.weights = averaged_weights_path
        final_train_opt.epochs = 20
        final_train_opt.name = 'exp_final_20_epochs'
        
        final_save_dir = Path(increment_path(Path(final_train_opt.project) / final_train_opt.name, exist_ok=final_train_opt.exist_ok))
        final_train_opt.save_dir = str(final_save_dir)

        train(final_train_opt.hyp, final_train_opt, device, callbacks)
        print(f"Final training complete.")

    finally:
        # IMPORTANT: Restore the original function to avoid side effects elsewhere.
        torch.load = _original_torch_load
        print("\nRestored original torch.load function.")
        print("\nProcess finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 Pruning, Retraining, and Averaging Script")
    # --- IMPORTANT: You must create a data.yaml file for your dataset ---
    # Example `signature_data.yaml`:
    # train: ../SignatureData/train/images
    # val: ../SignatureData/valid/images
    #
    # nc: 2  # number of classes
    # names: ['genuine', 'forged']  # class names
    parser.add_argument('--data', type=str, default='data/signature_data.yaml', help='path to your data.yaml file')
    parser.add_argument('--initial-weights', type=str, default='yolov5s.pt', help='initial weights path (e.g., yolov5s.pt, yolov5m.pt)')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path (optional)')
    parser.add_argument('--img-size', type=int, default=640, help='image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    parser.add_argument('--project', default='weights', help='save directory for pruned/averaged weights')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')

    opt = parser.parse_args()
    
    # Display the current time and location
    print("--- Script Start ---")
    print("Current Time: Wednesday, July 9, 2025 at 9:31 AM +06")
    print("Location: Dhaka, Dhaka Division, Bangladesh")
    print("--------------------")

    main(opt)
