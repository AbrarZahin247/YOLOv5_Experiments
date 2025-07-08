import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import yaml

# PyTorch 2.6 compatible attempt_load function
def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads a YOLOv5 model from a checkpoint file.

    Args:
        weights (str or Path): Path to the model weights file (.pt).
        device (torch.device, optional): The device to load the model onto. Defaults to None.
        inplace (bool, optional): Whether to modify the model in-place. Defaults to True.
        fuse (bool, optional): Whether to fuse Conv2d + BatchNorm2d layers. Defaults to True.

    Returns:
        (torch.nn.Module): The loaded PyTorch model.
    """
    model_path = Path(weights)
    if not model_path.exists():
        raise FileNotFoundError(f"'{weights}' does not exist")

    # Load the checkpoint
    ckpt = torch.load(weights, map_location=device)
    
    # Create the model from the 'model' key in the checkpoint
    # The 'model' in the checkpoint is often a deepcopy of the actual model
    # or a model definition that can be instantiated
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
        
    # If inplace, the model is modified directly.
    # Otherwise, a copy is made.
    if not inplace:
        model = model.copy()

    # Log a summary of the loaded model
    logging.info(f'Model summary: {len(list(model.modules()))} modules, {sum(p.numel() for p in model.parameters())} parameters')

    return model

# --- Utility functions from your script ---

def get_model_weights_as_vector(model):
    """Extracts and flattens all model weights into a single vector."""
    weights_vector = []
    for param in model.parameters():
        if param.requires_grad:
            weights_vector.append(param.data.clone().cpu().numpy().flatten())
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

    Args:
        model: The PyTorch model to modify.
        percentile (int): The percentile of weights to zero out (e.g., 90 for top 10%).
    """
    weights = get_model_weights_as_vector(model)
    threshold = np.percentile(np.abs(weights), percentile)
    weights[np.abs(weights) > threshold] = 0
    set_model_weights_from_vector(model, weights)
    print(f"Zeroed out weights above the {percentile}th percentile (magnitude > {threshold:.4f})")
    
# --- Main execution logic ---
# Note: The 'train' function is assumed to be in a 'train.py' file as in the original script.
# If 'train.py' is not available, the training steps will fail.
# You would need to have the YOLOv5 training infrastructure available.
try:
    from train import train
except ImportError:
    print("Warning: 'train' function could not be imported. The training steps will fail.")
    print("Please ensure this script is run within a YOLOv5 repository environment.")
    train = None

def main(opt):
    """Main function to run the YOLOv5 training process."""
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Step 1: Load a pre-trained YOLOv5 model ---
    print("Step 1: Loading pre-trained model...")
    model = attempt_load(opt.initial_weights, device=device)
    print("Model loaded successfully.")

    # --- Step 2: Zero out the 10% of weights with the highest magnitude ---
    print("\nStep 2: Zeroing out top 10% of weights by magnitude...")
    zero_top_weights(model, percentile=90)

    # --- Step 3: Save the pruned model weights ---
    pruned_weights_path = os.path.join(opt.project, 'yolov5s_pruned.pt')
    print(f"\nStep 3: Saving pruned model weights to {pruned_weights_path}...")
    os.makedirs(opt.project, exist_ok=True)
    save_dict = {'model': model, 'optimizer': None, 'epoch': -1}
    torch.save(save_dict, pruned_weights_path)
    print("Pruned model weights saved.")
    
    if not train:
        print("\nSkipping training steps because 'train' function is not available.")
        return

    # --- Step 4: Retrain the pruned model for 10 epochs ---
    print("\nStep 4: Retraining the pruned model for 10 epochs...")
    hyp_path = 'data/hyps/hyp.scratch-low.yaml'
    # Fallback if model.yaml is not available in the loaded model
    cfg_path = opt.cfg or (model.yaml['cfg'] if hasattr(model, 'yaml') and 'cfg' in model.yaml else '')

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
        evolve=None,
        bucket='',
        cache=opt.cache,
        image_weights=False,
        device=device,
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
        freeze=0,
        save_period=-1,
        local_rank=-1,
        entity=None,
        upload_dataset=False,
        bbox_interval=-1,
        artifact_alias="latest"
    )
    last_weights_path_10_epochs, _ = train(retrain_opt)
    print(f"Retraining complete. Weights saved after 10 epochs.")

    # --- Step 5: Average the pruned and retrained weights ---
    print("\nStep 5: Averaging pruned and retrained weights...")
    
    # The last weights are usually saved in a path like 'runs/train/exp_retrain_10_epochs/weights/last.pt'
    # We construct this path based on the training options
    last_weights_10_epochs = Path(retrain_opt.project) / retrain_opt.name / 'weights' / 'last.pt'

    pruned_model = torch.load(pruned_weights_path, map_location=device)['model']
    retrained_model = torch.load(last_weights_10_epochs, map_location=device)['model']

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
    print("\nStep 6: Continuing training with averaged weights for 20 epochs...")
    final_train_opt = retrain_opt
    final_train_opt.weights = averaged_weights_path
    final_train_opt.epochs = 20
    final_train_opt.name = 'exp_final_20_epochs'
    
    final_weights, _ = train(final_train_opt)
    print(f"Final training complete. Final model weights saved.")
    print("\nProcess complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/signature_data.yaml', help='path to your data.yaml file')
    parser.add_argument('--initial-weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--project', default='weights', help='save directory for pruned/averaged weights')
    parser.add_argument('--cache', action='store_true', help='cache images for faster training')

    opt = parser.parse_args()

    # This script assumes you are running it from the root of a yolov5 repository.
    # Ensure all dependencies for train.py are installed.
    main(opt)
