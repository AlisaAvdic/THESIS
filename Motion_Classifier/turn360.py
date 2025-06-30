# Superseded by time_based_360_detector.py, read the paper for comparison.
import os
import numpy as np
import math
import matplotlib.pyplot as plt

def angle_between(p1, p2):
    """
    Calculate the angle (in radians) between two 2D points.

    Parameters:
    p1 (tuple or list): Coordinates of the first point (x1, y1).
    p2 (tuple or list): Coordinates of the second point (x2, y2).

    Returns:
    float: The angle in radians between the two points, measured counterclockwise
           from the positive x-axis.
    """
    dx = p2[0] - p1[0]  # Difference in x-coordinates
    dy = p2[1] - p1[1]  # Difference in y-coordinates
    return math.atan2(dy, dx)  # Compute the angle using atan2

def detect_turns_with_cooldown(npy_folder, joint1_idx=2, joint2_idx=5,
                                window_size=15, rotation_thresh=300, cooldown_frames=20):
    files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')], key=lambda x: int(x.replace('.npy', '')))
    angles = []

    for f in files:
        data = np.load(os.path.join(npy_folder, f))
        if data.shape[0] <= max(joint1_idx, joint2_idx):
            angles.append(None)
            continue
        p1 = data[joint1_idx]
        p2 = data[joint2_idx]
        if np.all(p1 == 0) or np.all(p2 == 0):
            angles.append(None)
            continue
        angle = angle_between(p1, p2)
        angles.append(angle)

    angles = np.array([a if a is not None else np.nan for a in angles])
    unwrapped = np.unwrap(angles)

    detected_turns = []
    i = 0
    while i < len(unwrapped) - window_size:
        window = unwrapped[i:i+window_size]
        delta = np.diff(window)
        total_rotation = np.nansum(np.abs(delta))
        degrees_rotated = math.degrees(total_rotation)

        if degrees_rotated > rotation_thresh:
            detected_turns.append((i, i + window_size))
            i += window_size + cooldown_frames
        else:
            i += 1

    return detected_turns

def count_and_split_turns(npy_folder, joint1_idx=2, joint2_idx=5, plot=True):
    files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')], key=lambda x: int(x.replace('.npy', '')))
    file_to_index = {int(f.replace('.npy', '')): f for f in files}

    all_turns = []
    angles_lookup = {}

    for frame_num, fname in file_to_index.items():
        path = os.path.join(npy_folder, fname)
        data = np.load(path)
        if data.shape[0] <= max(joint1_idx, joint2_idx):
            continue
        p1 = data[joint1_idx]
        p2 = data[joint2_idx]
        if np.all(p1 == 0) or np.all(p2 == 0):
            continue
        angle = angle_between(p1, p2)
        angles_lookup[frame_num] = angle

    full_frames = sorted(angles_lookup.keys())
    full_angles = np.unwrap([angles_lookup[f] for f in full_frames])

    # Automatically detect motion windows first
    detected_ranges = detect_turns_with_cooldown(npy_folder, joint1_idx, joint2_idx)

    for (start, end) in detected_ranges:
        sub_frames = [f for f in full_frames if start <= f <= end]
        if len(sub_frames) < 2:
            continue

        segment_angles = np.unwrap([angles_lookup[f] for f in sub_frames])
        rel_segment = segment_angles - segment_angles[0]
        rotation_deg = np.degrees(rel_segment)

        curr_turn_start = sub_frames[0]
        prev_angle = 0
        last_split_idx = 0

        for i in range(1, len(rotation_deg)):
            delta = rotation_deg[i] - prev_angle
            if delta >= 360:
                curr_turn_end = sub_frames[i]
                all_turns.append((curr_turn_start, curr_turn_end))
                curr_turn_start = curr_turn_end
                prev_angle = rotation_deg[i]
                last_split_idx = i
            else:
                prev_angle = rotation_deg[i]

        if last_split_idx < len(sub_frames) - 1:
            all_turns.append((curr_turn_start, sub_frames[-1]))

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(full_frames, np.degrees(full_angles), label='Unwrapped Body Angle (°)')
        for (start, end) in all_turns:
            plt.axvspan(start, end, color='lime', alpha=0.3)
        plt.title('Automatically Detected 360° Turns')
        plt.xlabel('Frame')
        plt.ylabel('Angle (°)')
        plt.grid(True)
        plt.legend()
        plt.show(block=False)

    return all_turns

if __name__ == "__main__":
    # Can be replaced by any video skeletal data path:
    npy_folder_path = '/Users/alisaavdic/Desktop/THESIS/l2d_eval/ballet/I_15/data'
    split_turns = count_and_split_turns(npy_folder=npy_folder_path)

    print("\nGeneral 360° Turn Detection:")
    for i, (start, end) in enumerate(split_turns):
        print(f"  • Turn {i+1}: Frames {start} to {end}")

