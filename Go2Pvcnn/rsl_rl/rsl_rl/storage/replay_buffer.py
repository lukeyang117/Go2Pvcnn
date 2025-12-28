"""Replay buffer for storing point cloud and semantic label data for PVCNN training."""

import torch
import os
from typing import Optional, Tuple


class ReplayBuffer:
    """
    Replay buffer for storing point cloud and semantic label pairs.
    Used for asynchronous PVCNN finetuning separate from PPO training.
    
    Data is stored on disk to avoid memory issues with large point clouds.
    """
    
    def __init__(
        self,
        buffer_dir: str,
        max_size: int = 10000,
        device: str = "cpu"
    ):
        """
        Args:
            buffer_dir: Directory to store buffer data on disk
            max_size: Maximum number of samples to store
            device: Device for tensor operations
        """
        self.buffer_dir = buffer_dir
        self.max_size = max_size
        self.device = device
        
        # Create buffer directory if it doesn't exist
        os.makedirs(buffer_dir, exist_ok=True)
        
        # Circular buffer index
        self.current_idx = 0
        self.size = 0
        
        print(f"[ReplayBuffer] Initialized with max_size={max_size}, buffer_dir={buffer_dir}")
    
    def add(self, point_clouds: torch.Tensor, semantic_labels: torch.Tensor):
        """
        Add batch of samples to replay buffer.
        
        Args:
            point_clouds: (batch_size, num_points, 3) - XYZ coordinates
            semantic_labels: (batch_size, num_points) - Semantic class labels
        """
        batch_size = point_clouds.shape[0]
        
        for i in range(batch_size):
            # Save each sample as a separate file
            save_idx = self.current_idx % self.max_size
            save_path = os.path.join(self.buffer_dir, f"sample_{save_idx:06d}.pt")
            
            # Save to disk
            torch.save({
                'point_cloud': point_clouds[i].cpu(),
                'semantic_labels': semantic_labels[i].cpu()
            }, save_path)
            
            self.current_idx += 1
            self.size = min(self.size + 1, self.max_size)
        
        if self.current_idx % 1000 == 0:
            print(f"[ReplayBuffer] Added {self.current_idx} total samples, buffer size: {self.size}/{self.max_size}")
    
    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample random batch from replay buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            point_clouds: (batch_size, num_points, 3)
            semantic_labels: (batch_size, num_points)
            Returns None if buffer is empty
        """
        if self.size == 0:
            return None
        
        # Sample random indices
        indices = torch.randint(0, self.size, (batch_size,))
        
        # Load samples from disk
        point_clouds = []
        semantic_labels = []
        
        for idx in indices:
            load_path = os.path.join(self.buffer_dir, f"sample_{idx:06d}.pt")
            
            if os.path.exists(load_path):
                data = torch.load(load_path)
                point_clouds.append(data['point_cloud'])
                semantic_labels.append(data['semantic_labels'])
            else:
                # File doesn't exist (shouldn't happen), skip
                print(f"[ReplayBuffer] WARNING: Sample file {load_path} not found")
                continue
        
        if len(point_clouds) == 0:
            return None
        
        # Stack into batches and move to device
        point_clouds = torch.stack(point_clouds, dim=0).to(self.device)
        semantic_labels = torch.stack(semantic_labels, dim=0).to(self.device)
        
        return point_clouds, semantic_labels
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def clear(self):
        """Clear all data from replay buffer."""
        # Remove all files
        for i in range(self.size):
            file_path = os.path.join(self.buffer_dir, f"sample_{i:06d}.pt")
            if os.path.exists(file_path):
                os.remove(file_path)
        
        self.current_idx = 0
        self.size = 0
        print(f"[ReplayBuffer] Cleared buffer")
