import os
import sys
import pandas as pd
import argparse
import numpy as np
import glob
from pathlib import Path

# Get the project root directory (where orchestrator.py is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Add paths relative to project root
sys.path.extend([
    str(PROJECT_ROOT / 'Transpondancer' / 'src' / 'Ballet'),
    str(PROJECT_ROOT / 'Motion_Classifier'),
    str(PROJECT_ROOT / 'Ontology_Reasoner'),
    str(PROJECT_ROOT)
])

# Import the necessary modules from Transpondancer, Motion Classifier and Ontology Reasoner, resp.
import infer
import time_based_360_detector as motion_detector
from Ontology_Reasoner.ontology_reasoner import BalletOntologyReasoner, build_ballet_ontology

def run_pipeline(video_path, npy_folder, output_dir, ontology_path=None, frame_skip=0, show_graphs=True, write_graphs=False):
    """
    Run the complete ballet analysis pipeline.
    
    Args:
        video_path: Path to the video file
        npy_folder: Path to folder with .npy skeletal data files
        output_dir: Directory to save outputs
        ontology_path: Path to Ballet.owl (will build one if None)
        frame_skip: Process every nth frame from video
        show_graphs: Whether to display visualizations (default: True)
        write_graphs: Whether to write visualizations to files (default: False)
    """
    # Extract video name for output file naming
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Debugging information to verify correct video is being processed
    print(f"\n[DEBUG] Processing video: {video_path}")
    print(f"[DEBUG] Video name extracted: {video_name}")
    print(f"[DEBUG] NPY folder: {npy_folder}")
    
    # Create output directory if not yet exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Create video-specific output directory for graphs if write_graphs is True
    video_output_dir = os.path.join(output_dir, video_name)
    if write_graphs:
        os.makedirs(video_output_dir, exist_ok=True)
    
    # Default ontology path if not provided
    if not ontology_path:
        ontology_path = os.path.join(output_dir, 'Ballet.owl')
    
    # Step 1: Run pose classifier on video
    pose_csv_path = os.path.join(output_dir, f'{video_name}_pose_classifications.csv')
    print(f"\nStep 1: Classifying poses from video {video_path}...")
    
    # DEBUG: Check if the output file exists already and what frames it contains
    if os.path.exists(pose_csv_path):
        print(f"[DEBUG] Found existing pose CSV: {pose_csv_path}")
        try:
            existing_poses = pd.read_csv(pose_csv_path)
            print(f"[DEBUG] Existing pose file contains {len(existing_poses)} frames")
            print(f"[DEBUG] First few frames: {existing_poses['frame'].head().tolist()}")
            print(f"[DEBUG] Last few frames: {existing_poses['frame'].tail().tolist()}")
        except Exception as e:
            print(f"[DEBUG] Error reading existing CSV: {str(e)}")
    
    infer.classify_video_for_reasoner(video_path=video_path, output_csv_path=pose_csv_path, frame_skip=frame_skip)

    # DEBUG: Verify the pose classification output
    try:
        pose_df = pd.read_csv(pose_csv_path)
        print(f"[DEBUG] Generated pose classifications for {len(pose_df)} frames")
        print(f"[DEBUG] Frame range: {pose_df['frame'].min()} to {pose_df['frame'].max()}")
        pose_counts = pose_df['pose'].value_counts()
        print(f"[DEBUG] Pose distribution: {dict(pose_counts)}")
    except Exception as e:
        print(f"[DEBUG] Error reading generated CSV: {str(e)}")
    
    # Step 2: Run motion detector on skeletal data
    print(f"\nStep 2: Detecting motion patterns from {npy_folder}...")
    
    # DEBUG: Check what NPY files exist
    npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))
    print(f"[DEBUG] Found {len(npy_files)} NPY files in {npy_folder}")
    if npy_files:
        print(f"[DEBUG] First few NPY files: {[os.path.basename(f) for f in npy_files[:5]]}")
    
    motion_results = motion_detector.detect_turns_in_video(
        npy_folder=npy_folder, 
        fps=30, 
        plot=show_graphs, 
        save_plot=write_graphs,
        plot_path=os.path.join(video_output_dir, f'{video_name}_motion_detection.png') if write_graphs else None
    )
    
    # DEBUG: Show detected motion segments
    print(f"[DEBUG] Detected {len(motion_results)} motion segments")
    for idx, motion in enumerate(motion_results):
        motion_type = motion.get('type', 'Turn360')
        print(f"[DEBUG] Motion {idx+1}: {motion_type} from frame {motion['start_frame']} to {motion['end_frame']}")
    
    # Step 3: Run ontology reasoner 
    print("\nStep 3: Performing ontology reasoning...")
    
    # Create ontology if it doesn't exist
    if not os.path.exists(ontology_path):
        build_ballet_ontology(ontology_path)
    
    # Initialize reasoner and process results
    reasoner = BalletOntologyReasoner(ontology_path)
    pose_results = reasoner.load_pose_classifications(pose_csv_path)
    
    # DEBUG: Check what was loaded into the reasoner
    print(f"[DEBUG] Loaded {len(pose_results)} poses into reasoner")
    if not pose_results.empty:
        print(f"[DEBUG] First pose: {pose_results.iloc[0].to_dict()}")
        print(f"[DEBUG] Last pose: {pose_results.iloc[-1].to_dict()}")
    
    compound_movements = reasoner.infer_compound_movements(pose_results, motion_results)
    
    # DEBUG: Show inferred movements before generating outputs
    print(f"[DEBUG] Inferred {len(compound_movements)} compound movements")
    for idx, movement in enumerate(compound_movements):
        movement_type = movement.get('movement_type', 'Unknown')
        poses = [p.get('pose_type', 'Unknown') for p in movement.get('poses', [])]
        motions = [m.get('motion_type', 'Unknown') for m in movement.get('motions', [])]
        print(f"[DEBUG] Movement {idx+1}: Type={movement_type}, Poses={poses}, Motions={motions}")
    
    # Generate output files
    srt_content = reasoner.generate_srt_subtitles(compound_movements)
    srt_path = os.path.join(output_dir, f'{video_name}_ballet_movements.srt')
    with open(srt_path, 'w') as f:
        f.write(srt_content)
    
    # Save JSON results
    import json
    json_path = os.path.join(output_dir, f'{video_name}_ballet_movements.json')
    with open(json_path, 'w') as f:
        # Convert numpy values to Python native types before serializing to JSON
        serialized_movements = []
        for movement in compound_movements:
            serialized_movement = {}
            for key, value in movement.items():
                # Convert numpy integers to Python integers
                if isinstance(value, np.integer):
                    serialized_movement[key] = int(value)
                # Convert numpy floats to Python floats
                elif isinstance(value, np.floating):
                    serialized_movement[key] = float(value)
                else:
                    serialized_movement[key] = value
            serialized_movements.append(serialized_movement)
        json.dump(serialized_movements, f, indent=2)
    
    # Visualize results if requested
    if show_graphs or write_graphs:
        vis_path = os.path.join(video_output_dir, f'{video_name}_timeline.png') if write_graphs else None
        reasoner.visualize_results(
            compound_movements,
            show=show_graphs,
            save_path=vis_path
        )
        # Generate and visualize the turn timeline
        reasoner.visualize_turn_timeline(
            compound_movements,
            show=show_graphs,
            save_path=vis_path
        )
    
    # Print summary
    print(f"\nPipeline complete for {video_name}!")
    print(f"OK: Classified poses saved to: {pose_csv_path}")
    print(f"OK: Detected {len(motion_results)} motion segments")
    print(f"OK: Identified {len(compound_movements)} compound ballet movements")
    print(f"OK: SRT subtitles saved to: {srt_path}")
    print(f"OK: Detailed results saved to: {json_path}")
    if write_graphs:
        print(f"OK: Visualizations saved to: {video_output_dir}")
    
    return compound_movements


def process_directory(directory_path, output_dir, ontology_path, frame_skip=5, show_graphs=False, write_graphs=True):
    """
    Process all videos in a directory and its immediate subdirectories.
    
    Args:
        directory_path: Path to the directory containing videos
        output_dir: Directory to save outputs
        ontology_path: Path to Ballet.owl
        frame_skip: Process every nth frame from video
        show_graphs: Whether to display visualizations
        write_graphs: Whether to write visualizations to files
    """
    print(f"Searching for videos in {directory_path}...")
    
    # Find all video files in the directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    
    # Search in the main directory
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(directory_path, f'*{ext}')))
    
    # Search one level deep
    for subdir in [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]:
        subdir_path = os.path.join(directory_path, subdir)
        for ext in video_extensions:
            videos.extend(glob.glob(os.path.join(subdir_path, f'*{ext}')))
    
    if not videos:
        print("No videos found in the specified directory.")
        return
    
    print(f"Found {len(videos)} videos to process.")
    
    # Process each video
    for video_path in videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        
        # Look for data folder with .npy files
        npy_folder = os.path.join(video_dir, video_name, 'data')
        if not os.path.exists(npy_folder):
            npy_folder = os.path.join(video_dir, 'data')
        
        if not os.path.exists(npy_folder):
            print(f"WARNING: Could not find .npy folder for {video_path}. Skipping.")
            continue
        
        print(f"\n{'='*80}\nProcessing video: {video_path}\nNPY folder: {npy_folder}\n{'='*80}")
        
        try:
            run_pipeline(
                video_path=video_path,
                npy_folder=npy_folder,
                output_dir=output_dir,
                ontology_path=ontology_path,
                frame_skip=frame_skip,
                show_graphs=show_graphs,
                write_graphs=write_graphs
            )
        except Exception as e:
            print(f"ERROR processing {video_path}: {str(e)}")
            continue
    
    print(f"\nCompleted processing all videos in {directory_path}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ballet Movement Analysis Pipeline')
    parser.add_argument('--no-graphs', action='store_true', help='Run without displaying graphs')
    parser.add_argument('--write-graphs', action='store_true', help='Write graphs to files in output directory')
    parser.add_argument('--video-path', type=str, default="./I_17/I_17.mp4", help='Path to the video file to analyze or directory containing videos')
    parser.add_argument('--npy-folder', type=str, default="./I_17/data", help='Path to the folder containing skeletal data (.npy files)')
    parser.add_argument('--output-dir', type=str, default="./output", help='Directory to save results')
    parser.add_argument('--ontology-path', type=str, default="./Ontology_Reasoner/myBallet.owl", help='Path to the Ballet.owl ontology file')
    parser.add_argument('--frame-skip', type=int, default=5, help='Process every nth frame for pose classification (default: 5)')
    parser.add_argument('--batch', action='store_true', help='Process all videos in the directory specified by --video-path')
    
    args = parser.parse_args()
    
    # Check if we're processing a directory or a single video
    if args.batch or os.path.isdir(args.video_path):
        process_directory(
            directory_path=args.video_path,
            output_dir=args.output_dir,
            ontology_path=args.ontology_path,
            frame_skip=args.frame_skip,
            show_graphs=not args.no_graphs,
            write_graphs=args.write_graphs
        )
    else:
        # Run the pipeline with the specified arguments for a single video
        run_pipeline(
            video_path=args.video_path, 
            npy_folder=args.npy_folder, 
            output_dir=args.output_dir,
            ontology_path=args.ontology_path,
            frame_skip=args.frame_skip,
            show_graphs=not args.no_graphs,
            write_graphs=args.write_graphs
        )
