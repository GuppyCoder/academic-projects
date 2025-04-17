import argparse
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from homework.models import MODEL_FACTORY, save_model
from homework.datasets.road_dataset import load_data  # Use load_data instead of RoadDataset
from homework.metrics import PlannerMetric

# Set random seed for reproducibility.
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def compute_custom_loss(preds: torch.Tensor, waypoints: torch.Tensor, mask: torch.Tensor,
                        long_weight: float = 0.6, lat_weight: float = 0.4) -> torch.Tensor:
    """
    Compute a weighted L1 loss that treats the longitudinal (coordinate 0)
    and lateral (coordinate 1) errors differently.
    """
    # Compute absolute errors for each coordinate.
    loss_long = torch.abs(preds[..., 0] - waypoints[..., 0])
    loss_lat  = torch.abs(preds[..., 1] - waypoints[..., 1])
    # Combine losses with respective weights.
    loss_per_waypoint = long_weight * loss_long + lat_weight * loss_lat
    # Apply the mask and average over valid waypoints.
    loss = (loss_per_waypoint * mask.float()).sum() / mask.float().sum()
    return loss

def train(
    model_name: str,
    transform_pipeline: str,
    num_workers: int,
    lr: float,
    batch_size: int,
    num_epoch: int,
    dataset_path: str = "drive_data/train",  # Set the dataset path to your train folder.
):
    # Select device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Create a DataLoader using the load_data helper.
    dataloader = load_data(
        dataset_path=dataset_path,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True
    )

    # Instantiate the model.
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model name: {model_name}")
    model = MODEL_FACTORY[model_name]()  # Additional kwargs can be passed if needed.
    model.to(device)

    # Define optimizer with optional weight decay.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler: decrease LR every 10 epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize metric tracker.
    metric = PlannerMetric()

    best_l1_error = float('inf')
    best_model_state = None

    for i, batch in enumerate(dataloader):
      print("Batch keys:", list(batch.keys()))
      # Optionally, print the shapes or types:
      for key, value in batch.items():
          print(f"Key: {key}, Shape: {value.shape if hasattr(value, 'shape') else type(value)}")
      # Break after first batch for debugging.
      break



    # Training loop.
    model.train()
    start_time = time.time()
    for epoch in range(1, num_epoch + 1):
        running_loss = 0.0
        metric.reset()
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()
            
            # For CNNPlanner, use 'image'; otherwise use lane boundaries.
            if model_name == "cnn_planner":
                image = batch["image"].to(device)  # shape: (B, 3, 96, 128)
                preds = model(image=image)
            else:
                track_left = batch["track_left"].to(device)      # shape: (B, n_track, 2)
                track_right = batch["track_right"].to(device)      # shape: (B, n_track, 2)
                preds = model(track_left=track_left, track_right=track_right)
            
            waypoints = batch["waypoints"].to(device)          # shape: (B, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # shape: (B, n_waypoints)

            # Use the custom loss function.
            loss = compute_custom_loss(preds, waypoints, waypoints_mask, long_weight=0.6, lat_weight=0.4)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            metric.add(preds.detach().cpu(), waypoints.detach().cpu(), waypoints_mask.detach().cpu())

        avg_loss = running_loss / num_batches
        computed_metric = metric.compute()
        elapsed = time.time() - start_time

        # Checkpoint: update best model if L1 error improves.
        if computed_metric['l1_error'] < best_l1_error:
            best_l1_error = computed_metric['l1_error']
            best_model_state = model.state_dict()

        print(
            f"Epoch [{epoch}/{num_epoch}] - Loss: {avg_loss:.4f} - "
            f"L1: {computed_metric['l1_error']:.4f} "
            f"(Long: {computed_metric['longitudinal_error']:.4f}, Lat: {computed_metric['lateral_error']:.4f}) - "
            f"Time: {elapsed:.2f}s"
        )

        # Step the learning rate scheduler.
        scheduler.step()

    # Load best model state before saving.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the trained model.
    saved_path = save_model(model)
    print(f"Model saved to {saved_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model_name", type=str, default="mlp_planner", help="Model name to train")
    parser.add_argument("--transform_pipeline", type=str, default="state_only", help="Dataset transform pipeline")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epoch", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--dataset_path", type=str, default="drive_data", help="Path to the drive dataset")

    args = parser.parse_args()
    train(
        model_name=args.model_name,
        transform_pipeline=args.transform_pipeline,
        num_workers=args.num_workers,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        dataset_path=args.dataset_path,
    )
