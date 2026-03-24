"""
Trajectory Encoder
Encode a trajectory [B, T, 2] → [B, 1, D] as a single token embedding.
Two-layer MLP: flatten → Linear(T*2, D) → GELU → Linear(D, D)
"""

import torch
import torch.nn as nn


class TrajectoryEncoder(nn.Module):
    """
    Encode trajectory [B, T, 2] → [B, 1, D] as a single token embedding.
    Two-layer MLP: flatten → Linear(T*2, D) → GELU → Linear(D, D)
    """

    def __init__(self, max_time_steps: int, hidden_dim: int):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(max_time_steps * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj: [B, T, 2] — normalized (x, y) coordinates per timestep
        Returns:
            [B, 1, D] — one token embedding per sample
        """
        B, T, _ = traj.shape
        assert traj.shape[2] == 2, f"Expected traj last dim=2 (x,y coords), got {traj.shape[2]}"
        # Pad or truncate to max_time_steps
        if T < self.max_time_steps:
            pad = torch.zeros(
                B, self.max_time_steps - T, 2,
                device=traj.device, dtype=traj.dtype
            )
            traj = torch.cat([traj, pad], dim=1)
        elif T > self.max_time_steps:
            traj = traj[:, :self.max_time_steps, :]
        x = traj.reshape(B, -1)   # [B, max_time_steps * 2]
        x = self.encoder(x)        # [B, D]
        return x.unsqueeze(1)      # [B, 1, D]
