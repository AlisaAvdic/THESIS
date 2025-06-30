"""
Time based 360 detector

Supports annotation of '360° turn' (rotation), future(?) '180° turn', 'jump', etc.

Expects as input a folder with .npy files, where each file contains joint position data.

The workflow calculates angles between specified joints and detects turns based on those angles.

Usage:
1. Set the `npy_folder_path` variable to the directory containing the .npy files for the video.
2. Call `detect_turns_in_video(npy_folder_path)` to detect turns and visualize the results.
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import timedelta

def angle_between(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

def load_angle_data(npy_folder, joint1_idx=2, joint2_idx=5):
    """
    Load angle data from .npy files
    
    Returns:
    tuple: (angles, frame_indices)
    """
    files = sorted(
        [f for f in os.listdir(npy_folder) if f.endswith('.npy')],
        key=lambda x: int(x.replace('.npy', ''))
    )

    angles = []
    frame_indices = []

    for f in files:
        frame_idx = int(f.replace('.npy', ''))
        data = np.load(os.path.join(npy_folder, f))

        if data.shape[0] <= max(joint1_idx, joint2_idx):
            continue

        p1 = data[joint1_idx]
        p2 = data[joint2_idx]

        if np.all(p1 == 0) or np.all(p2 == 0):
            continue

        angle = angle_between(p1, p2)
        angles.append(angle)
        frame_indices.append(frame_idx)

    return np.unwrap(np.array(angles)), np.array(frame_indices)

def detect_turns_time_based(angles, frame_indices, fps=30, 
                            min_turn_sec=0.3, max_turn_sec=1.2, 
                            rotation_thresh=350, cooldown_sec=0.1):
    """
    Core function to detect turns based on time parameters
    """
    min_frames = int(min_turn_sec * fps)
    max_frames = int(max_turn_sec * fps)
    cooldown_frames = int(cooldown_sec * fps)
    
    detected_turns = []
    i = 0
    
    while i < len(angles) - min_frames:
        found_turn = False
        # Try different window sizes from min to max
        for window_size in range(min_frames, min(max_frames + 1, len(angles) - i)):
            window = angles[i:i + window_size]
            delta = np.diff(window)
            total_rotation = np.nansum(np.abs(delta))
            degrees_rotated = math.degrees(total_rotation)
            
            # If valid turn found:
            if degrees_rotated > rotation_thresh:
                start_frame = frame_indices[i]
                end_frame = frame_indices[i + window_size - 1]
                duration = (end_frame - start_frame + 1) / fps  # +1 because inclusive
                
                detected_turns.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_frame / fps,
                    'end_time': end_frame / fps,
                    'duration': duration,
                    'rotation': degrees_rotated
                })
                
                i += window_size + cooldown_frames
                found_turn = True
                break
        
        if not found_turn:
            i += 1
            
    return detected_turns

def format_time(seconds):
    """Convert seconds to a formatted time string (MM:SS.ms)"""
    td = timedelta(seconds=seconds)
    minutes, seconds = divmod(td.seconds, 60)
    return f"{minutes:02d}:{seconds:02d}.{int(td.microseconds / 1000):03d}"

def detect_turns_in_video(npy_folder,
                         fps=30,
                         joint1_idx=2, joint2_idx=5,
                         min_turn_sec=0.3, 
                         max_turn_sec=1.2,
                         rotation_thresh=350,
                         consecutive_turns_thresh=355,
                         cooldown_sec=0.1,
                         plot=True,
                         save_plot=False,
                         plot_path=None):
    """
    Simple and robust turn detector based on time parameters.
    
    Args:
        npy_folder: Path to folder containing .npy skeletal data files
        fps: Frames per second of the original video
        joint1_idx, joint2_idx: Indices of the joints to calculate angles between
        min_turn_sec, max_turn_sec: Min and max duration for a turn segment
        rotation_thresh: Rotation threshold in degrees for detecting turns
        consecutive_turns_thresh: Threshold for combining consecutive turns
        cooldown_sec: Cooldown period between turn detections
        plot: Whether to display the plot interactively
        save_plot: Whether to save the plot to a file
        plot_path: Path where to save the plot (required if save_plot is True)
    """
    # Load angle data
    angles, frame_indices = load_angle_data(npy_folder, joint1_idx, joint2_idx)
    
    # Detect basic turns
    turns = detect_turns_time_based(
        angles, frame_indices, fps=fps,
        min_turn_sec=min_turn_sec, 
        max_turn_sec=max_turn_sec,
        rotation_thresh=rotation_thresh, 
        cooldown_sec=cooldown_sec
    )
    
    # Plot if requested
    if (plot or save_plot) and turns:
        plt.figure(figsize=(12, 6))
        plt.plot(frame_indices, np.degrees(angles), label='Unwrapped Angle (°)')
        
        # Calculate angular velocity for visualization
        if len(angles) > 1:
            angular_velocity = np.diff(np.degrees(angles)) * fps
            angular_velocity = np.append(angular_velocity, angular_velocity[-1])
            plt.plot(frame_indices, angular_velocity, '--', color='orange', alpha=0.5, label='Angular Velocity (°/s)')
        
        # Highlight detected turns
        for i, turn in enumerate(turns):
            start_frame = turn['start_frame']
            end_frame = turn['end_frame']
            duration = turn['duration']
            plt.axvspan(start_frame, end_frame, color='lime', alpha=0.3)
            plt.annotate(f"{duration:.2f}s", 
                         xy=((start_frame + end_frame) / 2, np.degrees(angles[frame_indices == start_frame][0])),
                         xytext=(0, 30), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->"), 
                         ha='center')
                         
        plt.title('Time-Based 360° Turn Detection')
        plt.xlabel(f'Frame (at {fps} fps)')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        # Save plot if parameter set
        if save_plot and plot_path:
            plt.savefig(plot_path, dpi=300)
            print(f"Turn detection plot saved to {plot_path}")
            
        # Show plot if parameter set
        if plot:
            plt.show(block=False)
        else:
            plt.close()
    
    # Format results for output
    formatted_turns = []
    for i, turn in enumerate(turns):
        formatted_turns.append({
            'id': i + 1,
            'start_frame': turn['start_frame'],
            'end_frame': turn['end_frame'],
            'start_time': turn['start_time'],
            'end_time': turn['end_time'],
            'duration_sec': turn['duration'],
            'start_time_formatted': format_time(turn['start_time']),
            'end_time_formatted': format_time(turn['end_time']),
            'rotation_degrees': round(turn['rotation'], 1),
            'type': 'Turn360'
        })
    
    return formatted_turns

def visualize_turn360_timeline(turns, max_frame=None, show=True, save_path=None):
    """
    Create a timeline visualization for 360° turns with consistent styling.
    
    Args:
        turns: List of detected turn segments
        max_frame: Maximum frame number (if None, computed from data)
        show: Whether to display the plot interactively
        save_path: Path where to save the visualization
    """
    if not turns:
        print("No turns to visualize")
        return
    
    # Calculate max_frame if not provided
    if max_frame is None:
        max_frame = max([turn['end_frame'] for turn in turns]) + 30
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 2))
    
    # Extract video name from save path
    video_name = "Unknown"
    if save_path:
        video_name = os.path.basename(save_path).split('_')[0]
    
    # Colors
    TURN360_COLOR = '#4169E1'  # Royal Blue
    
    # Draw turns on timeline
    for i, turn in enumerate(turns):
        ax.hlines(y=0, xmin=turn['start_frame'], xmax=turn['end_frame'],
                  color=TURN360_COLOR, linewidth=8, alpha=0.7)
    
    # Configure axes
    ax.set_yticks([0])
    ax.set_yticklabels(['Turn360'], fontsize=10)
    
    ax.set_xlabel('Frame Number', color='black')
    ax.set_xlim(0, max_frame)
    ax.set_ylim(-0.5, 0.5)
    
    # Title with video name
    ax.set_title(f'360° Turn Timeline for Video {video_name}', color='black', pad=10)
    
    # Frame number ticks
    frame_interval = max(5, max_frame // 20)
    frame_ticks = range(0, max_frame + 1, frame_interval)
    ax.set_xticks(frame_ticks)
    ax.tick_params(axis='both', colors='black')
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', color='black')
    
    # Set spines to black
    for spine in ax.spines.values():
        spine.set_color('black')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Turn360 timeline visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)

if __name__ == "__main__":
    # Can be replaced by any video skeletal data path:
    npy_folder_path = '/Users/alisaavdic/Desktop/THESIS/l2d_eval/ballet/I_17/data'
    video_fps = 30  # Specify the source video's frame rate
    
    turns = detect_turns_in_video(
        npy_folder=npy_folder_path,
        fps=video_fps,
        min_turn_sec=0.3,       # Minimum duration for a turn in seconds
        max_turn_sec=1.2,       # Maximum duration for a turn in seconds
        rotation_thresh=350,    # Rotation threshold in degrees (350 is just under a full turn)
        cooldown_sec=0.1        # Short cooldown between turns (100ms)
    )
    
    print("\nDetected Time-Denominated 360° Turns:")   
    for turn in turns:
        print(f"Turn {turn['id']}: Frames {turn['start_frame']} to {turn['end_frame']} "
              f"({turn['start_time_formatted']} - {turn['end_time_formatted']}, "
              f"{turn['duration_sec']:.2f} sec) - Rotation: {turn['rotation_degrees']}°")
        