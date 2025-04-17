from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,  # You can experiment with this value
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Each side: n_track points with 2 coords → total input dim = 2 * n_track * 2.
        input_dim = 2 * n_track * 2  # 2 * 10 * 2 = 40 by default

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout layer
            nn.Linear(hidden_dim, n_waypoints * 2)  # output: 3 waypoints * 2 = 6 values
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Concatenate boundaries along the track dimension: (B, 2*n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)
        # Flatten to shape (B, 2*n_track*2)
        x = x.view(x.size(0), -1)
        # Pass through the MLP
        out = self.model(x)
        # Reshape output to (B, n_waypoints, 2)
        out = out.view(x.size(0), self.n_waypoints, 2)
        return out


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,  # Explicit dropout parameter.
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Project each 2D lane boundary point into a d_model-dimensional feature.
        self.input_proj = nn.Linear(2, d_model)

        # Learned query embeddings: one per waypoint.
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer decoder (no attention mask required for this full-trajectory prediction).
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Map the Transformer output to 2D waypoint coordinates.
        self.fc_out = nn.Linear(d_model, 2)
        
    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Concatenate left and right boundaries: (B, 2*n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)
        # Project each 2D point into a feature space: (B, 2*n_track, d_model)
        x = self.input_proj(x)
        # Transformer modules expect sequence-first: (S, B, d_model) where S = 2*n_track
        x = x.transpose(0, 1)

        B = x.shape[1]
        # Get learned query embeddings: (n_waypoints, d_model)
        queries = self.query_embed.weight  # shape: (n_waypoints, d_model)
        # Expand queries to have batch dimension: (n_waypoints, B, d_model)
        queries = queries.unsqueeze(1).repeat(1, B, 1)

        # Use the Transformer decoder.
        out = self.transformer_decoder(tgt=queries, memory=x)  # shape: (n_waypoints, B, d_model)

        # Map the Transformer output to 2D coordinates: (n_waypoints, B, 2)
        out = self.fc_out(out)
        # Transpose back to (B, n_waypoints, 2)
        out = out.transpose(0, 1)
        return out


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        # Register buffers for input normalization.
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define a simple CNN backbone.
        # Input: (B, 3, 96, 128)
        self.conv_layers = nn.Sequential(
            # Block 1: (B, 3, 96, 128) -> (B, 16, 48, 64)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: (B, 16, 48, 64) -> (B, 32, 24, 32)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: (B, 32, 24, 32) -> (B, 64, 12, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # After conv layers, the feature map has shape (B, 64, 12, 16)
        # Flattened size: 64 * 12 * 16 = 12288

        # A small fully connected network to map features to waypoints.
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2)  # Predict n_waypoints * 2 values.
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        # Normalize the input.
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # Extract features using the CNN backbone.
        x = self.conv_layers(x)
        # Flatten the feature maps.
        x = x.view(x.size(0), -1)
        # Produce a (B, n_waypoints * 2) tensor.
        x = self.fc(x)
        # Reshape to (B, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)
        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
