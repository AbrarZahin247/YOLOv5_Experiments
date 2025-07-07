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

def main(opt):
    """Main function to run the YOLOv5 training process."""
    set_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Step 1: Load a pre-trained YOLOv5 model ---
    print("Step 1: Loading pre-trained model...")
    model = attempt_load(opt.initial_weights, map_location=device)
    print("Model loaded successfully.")

    # --- Step 2: Zero out the 10% of weights with the highest magnitude ---
    print("\nStep 2: Zeroing out top 10% of weights by magnitude...")
    zero_top_weights(model, percentile=90)

    # --- Step 3: Save the pruned model weights ---
    pruned_weights_path = os.path.join(opt.project, 'yolov5s_pruned.pt')
    print(f"\nStep 3: Saving pruned model weights to {pruned_weights_path}...")
    os.makedirs(opt.project, exist_ok=True)
    # Save the model correctly for YOLOv5 training
    save_dict = {'model': model.state_dict(), 'optimizer': None, 'epoch': -1}
    torch.save(save_dict, pruned_weights_path)
    print("Pruned model weights saved.")

    # --- Step 4: Retrain the pruned model for 10 epochs ---
    print("\nStep 4: Retraining the pruned model for 10 epochs...")
    # Setup options for the first retraining phase
    hyp_path = 'data/hyps/hyp.scratch-low.yaml'
    retrain_opt = argparse.Namespace(
        weights=pruned_weights_path,
        cfg=opt.cfg or model.yaml['cfg'],
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
        cache_images=opt.cache,
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
    # The train function returns the path to the last weights
    last_weights_10_epochs = train(retrain_opt)
    print(f"Retraining complete. Weights saved at {last_weights_10_epochs}")

    # --- Step 5: Average the pruned and retrained weights ---
    print("\nStep 5: Averaging pruned and retrained weights...")

    # Load the state dicts
    pruned_state_dict = torch.load(pruned_weights_path, map_location=device)['model']
    retrained_state_dict = torch.load(last_weights_10_epochs, map_location=device)['model']

    # Create a new model to hold the averaged weights
    averaged_model = attempt_load(opt.initial_weights, map_location=device)

    averaged_state_dict = averaged_model.state_dict()
    for key in pruned_state_dict:
        if key in retrained_state_dict and pruned_state_dict[key].data.dtype.is_floating_point:
            averaged_state_dict[key].data = (pruned_state_dict[key].data + retrained_state_dict[key].data) / 2.0
        else: # For non-floating point tensors or mismatched keys, copy from pruned
            averaged_state_dict[key].data = pruned_state_dict[key].data

    averaged_model.load_state_dict(averaged_state_dict)

    averaged_weights_path = os.path.join(opt.project, 'yolov5s_averaged.pt')
    avg_save_dict = {'model': averaged_model.state_dict(), 'optimizer': None, 'epoch': -1}
    torch.save(avg_save_dict, averaged_weights_path)
    print(f"Averaged model weights saved to {averaged_weights_path}")

    # --- Step 6: Continue training with averaged weights for 20 epochs ---
    print("\nStep 6: Continuing training with averaged weights for 20 epochs...")
    final_train_opt = retrain_opt
    final_train_opt.weights = averaged_weights_path
    final_train_opt.epochs = 20
    final_train_opt.name = 'exp_final_20_epochs'

    final_weights = train(final_train_opt)
    print(f"Final training complete. Final model weights saved at {final_weights}")
    print("\nProcess complete.")
