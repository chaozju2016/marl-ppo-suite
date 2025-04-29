import os
import time
import numpy as np
import imageio
import glob
import platform
from typing import List, Optional

def save_video(
    frames: List[np.ndarray],
    env_name: str,
    scene_name: str,
    algo: str,
    fps: int = 30,
    max_length: Optional[float] = 30,
    output_dir: str = "videos",
    quality: int = 8,
    codec: str = "libx264"
) -> Optional[str]:
    """
    Save frames as a video file in an organized directory structure.
    
    Args:
        frames: List of frames to save as video
        env_name: Name of the environment
        scene_name: Name of the map or scene
        algo: Name of the algorithm
        fps: Frames per second for the video. Defaults to 30.
        max_length: Maximum video length in seconds. If provided,
            frames will be sampled to fit within this duration.
        output_dir: Base directory for saving videos. Defaults to "videos".
        quality: Video quality (0-10, higher is better). Defaults to 8.
        codec: Video codec to use. Defaults to "libx264".
    
    Returns:
        str: Path to the saved video file, or None if saving failed
    """
    if not frames:
        print("No frames to save")
        return None
        
    try:
        # Validate and process frames
        frames = np.stack([
            frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
            for frame in frames
        ])
        
        # Sample frames if max_length is specified
        if max_length is not None and len(frames) > max_length * fps:
            target_frames = int(max_length * fps)
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = frames[indices]
        
        # Create nested directory structure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        videos_dir = os.path.join(output_dir, f"{env_name}_{scene_name}", algo)
        os.makedirs(videos_dir, exist_ok=True)
        
        # Create video path
        video_path = os.path.join(videos_dir, f"render_{timestamp}.mp4")
        
        # Configure video writer
        writer_kwargs = {
            "fps": fps,
            "quality": quality,
            "codec": codec,
            "pixelformat": "yuv420p",  # Ensure compatibility
            "macro_block_size": 16      # Prevent odd dimensions
        }
        
        # Save video
        with imageio.get_writer(video_path, **writer_kwargs) as writer:
            for frame in frames:
                writer.append_data(frame)
                
        print(f"Saved video to {video_path}")
        return video_path
        
    except Exception as e:
        print(f"Failed to save video: {e}")
        if frames is not None:
            print(f"Frame shape: {frames[0].shape}, dtype: {frames[0].dtype}")
        return None

def get_latest_sc2_replay():
    """
    Get the path to the most recently saved StarCraft II replay from the default replays folder.
    
    Returns:
        str: Path to the latest replay file, or None if no replays found
    """
    # Get SC2PATH from environment variable
    sc2_path = os.getenv('SC2PATH')
    
    if not sc2_path:
        # Default paths based on OS
        system = platform.system()
        if system == "Windows":
            sc2_path = os.path.join(os.environ['PROGRAMFILES'], 'StarCraft II')
        elif system == "Linux":
            sc2_path = os.path.expanduser('~/StarCraftII')
        elif system == "Darwin":  # macOS
            sc2_path = '/Applications/StarCraft II'

    #Path to replays directory
    replays_path = os.path.join(sc2_path, 'Replays')
    
    if not os.path.exists(replays_path):
        print(f"Replays directory not found at: {replays_path}")
        return None
        
    # Find all replay files (including those in subdirectories)
    replay_files = glob.glob(os.path.join(replays_path, '**', '*.SC2Replay'), recursive=True)
    
    if not replay_files:
        print("No replay files found")
        return None
        
    # Get the most recent replay
    latest_replay = max(replay_files, key=os.path.getctime)
    return latest_replay