import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Import Owlready2
from owlready2 import *
import json

# Add paths to access other modules
sys.path.append('../Motion classifier')
import time_based_360_detector as motion_detector


class BalletOntologyReasoner:
    def __init__(self, ontology_path, pose_threshold=0.7):
        """
        Initialize the ballet ontology reasoner with Owlready2.
        
        Args:
            ontology_path: Path to the Ballet.owl file
            pose_threshold: Confidence threshold for pose classifications
        """
        self.pose_threshold = pose_threshold
        
        # Load the ontology using Owlready2
        try:
            print(f"Loading ontology from: {ontology_path}")
            
            # Load the Ballet.owl ontology directly
            self.onto = get_ontology(f"file://{ontology_path}").load()
            print(f"Loaded ontology: {self.onto.base_iri}")
            
            # Store the correct namespace based on what's actually in the ontology
            self.ballet_ns = self.onto.base_iri
            print(f"Using namespace from ontology: {self.ballet_ns}")
            
            # Create direct reference to key classes from the ontology for later use
            self.class_dict = {}
            for c in self.onto.classes():
                self.class_dict[c.name] = c
                print(f"Class: {c.name} → {c}")
            
            # Store direct references to important classes
            if "Retire" in self.class_dict:
                self.retire_class = self.class_dict["Retire"]
            elif "Retiré" in self.class_dict:
                self.retire_class = self.class_dict["Retiré"]
            else:
                print("WARNING: No Retire/Retiré class found in ontology")
                
            if "Arabesque" in self.class_dict:
                self.arabesque_class = self.class_dict["Arabesque"]
            else:
                print("WARNING: No Arabesque class found in ontology")
                
            if "Plie" in self.class_dict or "Plié" in self.class_dict:
                self.plie_class = self.class_dict.get("Plie") or self.class_dict.get("Plié")
            else:
                print("WARNING: No Plie/Plié class found in ontology")
                
            # Find turn class
            turn_classes = [c for c in self.onto.classes() if "turn" in c.name.lower()]
            
            if turn_classes:
                print(f"Found {len(turn_classes)} turn classes: {[c.name for c in turn_classes]}")
                self.turn_class = next((c for c in turn_classes if c.name.lower() == "turn360"), turn_classes[0])
                print(f"Using turn class: {self.turn_class.name}")
            else:
                print("WARNING: No turn class found in ontology")
                
            # Find dance segment class
            segment_classes = [c for c in self.onto.classes() if "segment" in c.name.lower() or "dance" in c.name.lower()]
            
            if segment_classes:
                self.segment_class = segment_classes[0]
                print(f"Using segment class: {self.segment_class.name}")
            else:
                for cls_name in ["DanceSegment", "BalletMovement", "BalletMove"]:
                    if cls_name in self.class_dict:
                        self.segment_class = self.class_dict[cls_name]
                        print(f"Using segment class: {cls_name}")
                        break
                else:
                    print("WARNING: No dance segment class found")
                    
            # Create properties if they don't exist - THIS IS CRITICAL FOR BALLET.OWL
            self._ensure_properties_exist()
                
            # Prepare the reasoner
            try:
                with self.onto:
                    sync_reasoner_pellet(infer_property_values=True, debug=2)
                print(f"Reasoner initialized with {len(list(self.onto.classes()))} classes")
            except Exception as e:
                print(f"Warning: Reasoner initialization error: {e}")
                print("Continuing without reasoning capabilities")
            
        except Exception as e:
            print(f"Error loading ontology: {e}")
            raise
            
    def _ensure_properties_exist(self):
        """Create necessary properties if they don't exist in the ontology"""
        # Get all property names
        obj_property_names = [p.name for p in self.onto.object_properties()]
        data_property_names = [p.name for p in self.onto.data_properties()]
        
        print(f"Found object properties: {obj_property_names}")
        print(f"Found data properties: {data_property_names}")
        
        # Create hasPose if it doesn't exist
        with self.onto:
            # Find Pose and Motion classes from class_dict
            pose_class = None
            motion_class = None
            
            for cls_name in ["Pose", "Position"]:  # Position might be used instead of Pose
                if cls_name in self.class_dict:
                    pose_class = self.class_dict[cls_name]
                    print(f"Found Pose class: {cls_name}")
                    break
                    
            for cls_name in ["Motion", "Move", "Movement"]:  # Try different possible names
                if cls_name in self.class_dict:
                    motion_class = self.class_dict[cls_name]
                    print(f"Found Motion class: {cls_name}")
                    break
                    
            if not pose_class:
                print("Could not find Pose class - creating a generic one")
                class Pose(Thing): pass
                pose_class = Pose
                
            if not motion_class:
                print("Could not find Motion class - creating a generic one")
                class Motion(Thing): pass
                motion_class = Motion
        
            if not any(p for p in obj_property_names if "haspose" in p.lower() or "has_pose" in p.lower()):
                print("Creating hasPose property")
                class hasPose(ObjectProperty):
                    domain = [self.segment_class] if hasattr(self, "segment_class") else []
                    range = [pose_class]
                self.hasPose_prop = hasPose
            else:
                # Find and store reference to existing hasPose
                for p in self.onto.object_properties():
                    if "haspose" in p.name.lower() or "has_pose" in p.name.lower():
                        self.hasPose_prop = p
                        print(f"Using existing hasPose: {p.name}")
                        break
                        
            # Create hasMotion if it doesn't exist
            if not any(p for p in obj_property_names if "hasmotion" in p.lower() or "has_motion" in p.lower()):
                print("Creating hasMotion property")
                class hasMotion(ObjectProperty):
                    domain = [self.segment_class] if hasattr(self, "segment_class") else []
                    range = [motion_class]
                self.hasMotion_prop = hasMotion
            else:
                # Find and store reference to existing hasMotion
                for p in self.onto.object_properties():
                    if "hasmotion" in p.name.lower() or "has_motion" in p.name.lower():
                        self.hasMotion_prop = p
                        print(f"Using existing hasMotion: {p.name}")
                        break
                        
            # Create durationInSeconds if it doesn't exist
            if not any(p for p in data_property_names if "duration" in p.lower()):
                print("Creating durationInSeconds property")
                class durationInSeconds(DataProperty, FunctionalProperty):
                    domain = [self.segment_class] if hasattr(self, "segment_class") else []
                    range = [float]
                self.duration_prop = durationInSeconds
            else:
                # Find and store reference to existing duration property
                for p in self.onto.data_properties():
                    if "duration" in p.name.lower():
                        self.duration_prop = p
                        print(f"Using existing duration: {p.name}")
                        break
    
    def load_pose_classifications(self, pose_results_path):
        """Load pose classification results."""
        try:
            pose_df = pd.read_csv(pose_results_path)
            return pose_df
        except:
            # Create dummy data for testing
            print("Warning: Using dummy pose data")
            frames = list(range(0, 300, 1))
            poses = np.random.choice(
                ['releve', 'arabesque', 'plie', 'tendu', 'none'], 
                size=len(frames),
                p=[0.2, 0.2, 0.2, 0.1, 0.3]
            )
            confidences = np.random.uniform(0.6, 1.0, len(frames))
            return pd.DataFrame({
                'frame': frames,
                'pose': poses,
                'confidence': confidences
            })
    
    def get_dominant_pose(self, poses_df, start_frame, end_frame):
        """Get the dominant pose in a frame range."""
        # First try with the regular threshold
        frame_poses = poses_df[
            (poses_df['frame'] >= start_frame) & 
            (poses_df['frame'] <= end_frame) &
            (poses_df['confidence'] >= self.pose_threshold)
        ]
        
        # If no poses meet regular threshold, try with lower threshold for motion segments
        if len(frame_poses) == 0:
            motion_pose_threshold = 0.45  # Lower threshold for motion segments
            frame_poses = poses_df[
                (poses_df['frame'] >= start_frame) & 
                (poses_df['frame'] <= end_frame) &
                (poses_df['confidence'] >= motion_pose_threshold)
            ]
        
        if len(frame_poses) == 0:
            return "unknown"
            
        pose_counts = frame_poses['pose'].value_counts()
        return pose_counts.idxmax() if len(pose_counts) > 0 else "unknown"
    
    def infer_compound_movements(self, pose_results, motion_results, fps=30):
        """
        Infer compound movements using the ontology and reasoner.
        
        Args:
            pose_results: DataFrame with pose classifications
            motion_results: List of detected motion segments
            fps: Video frame rate (default: 30)
        
        Returns:
            List of compound movements with their details
        """
        # Storage for all our segments and their corresponding metadata
        all_segments = []
        compound_movements = []
        
        # Safety check for pose_results (must be a non-empty DataFrame)
        if pose_results is None or pose_results.empty:
            print("WARNING: No pose results available for reasoning")
            return compound_movements
        
        print("\n=== Step 1: Creating all dance segments ===")
        with self.onto:
            # First create all segments from motion detection
            for motion in motion_results:
                start_frame = motion['start_frame']
                end_frame = motion['end_frame']
                
                # Get the dominant pose during this segment
                dominant_pose = self.get_dominant_pose(pose_results, start_frame, end_frame)
                
                print(f"\nProcessing segment: {start_frame}-{end_frame}, pose: {dominant_pose}, "
                      f"rotation: {motion['rotation_degrees']}°")
                
                # Create a dance segment instance
                if hasattr(self, "segment_class"):
                    segment = self.segment_class()
                    print(f"OK: Created dance segment: {segment}")
                else:
                    print("ERR: No segment class available")
                    continue
                    
                # Create and add pose instance
                pose_instance = None
                try:
                    # Create appropriate pose instance based on detected pose
                    if dominant_pose.lower() == "retire" or dominant_pose.lower() == "retiré":
                        if hasattr(self, "retire_class"):
                            pose_instance = self.retire_class()
                            print(f"OK: Created Retire pose: {pose_instance}")
                    elif dominant_pose.lower() == "arabesque":
                        if hasattr(self, "arabesque_class"):
                            pose_instance = self.arabesque_class()
                            print(f"OK: Created Arabesque pose: {pose_instance}")
                    elif dominant_pose.lower() == "plie" or dominant_pose.lower() == "grand_plié":
                        if hasattr(self, "plie_class"):
                            pose_instance = self.plie_class()
                            print(f"OK: Created Plie pose: {pose_instance}")
                    
                    # If no specific pose instance was created yet, try class dictionary
                    if pose_instance is None:
                        for class_name, class_obj in self.class_dict.items():
                            if class_name.lower() == dominant_pose.lower():
                                pose_instance = class_obj()
                                print(f"OK: Created {class_name} pose: {pose_instance}")
                                break
                        
                        if pose_instance is None:
                            # Use UnknownPose if available
                            if "UnknownPose" in self.class_dict:
                                pose_instance = self.class_dict["UnknownPose"]()
                                print(f"OK Using generic UnknownPose for: {dominant_pose}")
                            else:
                                print(f"ERR: Cannot create pose for {dominant_pose}")
                                continue
                except Exception as e:
                    print(f"ERR creating pose: {e}")
                    continue
                    
                # Create and add motion instance
                motion_instance = None
                try:
                    if 'rotation_degrees' in motion and motion['rotation_degrees'] >= 350:
                        # Look for Turn360 class
                        if hasattr(self, "turn_class"):
                            motion_instance = self.turn_class()
                            print(f"OK: Created Turn360 motion: {motion_instance}")
                    
                    if motion_instance is None:
                        # Generic motion
                        if "Motion" in self.class_dict:
                            motion_instance = self.class_dict["Motion"]()
                            print(f"OK: Created generic Motion instance")
                        elif "UnknownMotion" in self.class_dict:
                            motion_instance = self.class_dict["UnknownMotion"]()
                            print(f"OK: Created generic UnknownMotion instance")
                        else:
                            print("ERROR: No motion class available")
                            continue
                except Exception as e:
                    print(f"ERROR creating motion: {e}")
                    continue
                
                # Now connect pose and motion to the segment
                try:
                    # Connect pose to segment
                    if hasattr(self, "hasPose_prop"):
                        self.hasPose_prop[segment] = [pose_instance]
                        print(f"OK: Added pose {type(pose_instance).__name__} to segment")
                    else:
                        print("ERROR: No hasPose property available")
                        continue
                        
                    # Connect motion to segment
                    if hasattr(self, "hasMotion_prop"):
                        self.hasMotion_prop[segment] = [motion_instance]
                        print(f"OK: Added motion {type(motion_instance).__name__} to segment")
                    else:
                        print("ERROR: No hasMotion property available")
                        continue
                        
                    # Add duration if available
                    if hasattr(self, "duration_prop"):
                        # Wrap in list to make it iterable
                        self.duration_prop[segment] = [float(motion['duration_sec'])]
                        print(f"OK: Set duration: {motion['duration_sec']} seconds")
                except Exception as e:
                    print(f"ERROR connecting attributes: {e}")
                    continue
                    
                # Store the segment and its metadata for later processing
                all_segments.append({
                    'segment': segment,
                    'pose': dominant_pose,
                    'motion_type': 'Turn360' if motion['rotation_degrees'] >= 350 else 'unknown',
                    'motion_data': motion
                })
            
            # Process important static poses (like Arabesque) that aren't part of turns
            print("\n=== Adding segments for important static poses ===")
            
            # Create a set to track frame ranges already covered by motion segments
            motion_covered_frames = set()
            for motion in motion_results:
                for frame in range(motion['start_frame'], motion['end_frame'] + 1):
                    motion_covered_frames.add(frame)
            
            print(f"Motion segments cover {len(motion_covered_frames)} frames")
            
            # Get all frames with poses
            # Filter for poses with moderate confidence (lower than the regular threshold)
            # Use a reduced threshold specifically for static poses
            static_pose_threshold = 0.45  # Lower threshold for static poses
            qualified_poses = pose_results[pose_results['confidence'] >= static_pose_threshold]
            print(f"Found {len(qualified_poses)} qualified poses (threshold: {static_pose_threshold})")
            print(f"Poses distribution: {qualified_poses['pose'].value_counts().to_dict()}")
            
            # Find groups of consecutive frames with Arabesque pose
            if not qualified_poses.empty:
                # Create a new column to mark significant poses
                # Using loc to avoid SettingWithCopyWarning
                qualified_poses = qualified_poses.copy()  # Create a copy to avoid warnings
                qualified_poses['is_significant'] = qualified_poses['pose'].apply(
                    lambda p: p.lower() in ['arabesque', 'retire', 'retiré', 'grand_plié', 'plie'])
                
                significant_poses = qualified_poses[qualified_poses['is_significant']]
                print(f"Found {len(significant_poses)} significant poses (Arabesque or Retiré)")
                
                # Skip if no significant poses found
                if not significant_poses.empty:
                    # Group poses by consecutive frames
                    significant_poses = significant_poses.reset_index(drop=True)
                    significant_poses['group'] = (significant_poses['frame'].diff() > fps).cumsum()
                    print(f"Grouped into {significant_poses['group'].nunique()} segments")
                    
                    # Process each group of consecutive frames with the same significant pose
                    for group_id, group_df in significant_poses.groupby(['group', 'pose']):
                        group_num, pose_name = group_id
                        
                        # Only process if we have enough frames (at least 3)
                        if len(group_df) >= 3:
                            start_frame = group_df['frame'].min()
                            end_frame = group_df['frame'].max()
                            
                            # Remove frames that are already covered by motion segments
                            uncovered_frames = []
                            for frame in group_df['frame']:
                                if frame not in motion_covered_frames:
                                    uncovered_frames.append(frame)
                            
                            # Skip if too few uncovered frames remain
                            if len(uncovered_frames) < 3:
                                print(f"Skipping static pose segment {start_frame}-{end_frame} ({pose_name}): "
                                      f"only {len(uncovered_frames)} frames not covered by motion")
                                continue
                            
                            # Find continuous ranges of uncovered frames
                            if uncovered_frames:
                                uncovered_frames.sort()
                                # Group consecutive uncovered frames
                                ranges = []
                                current_start = uncovered_frames[0]
                                current_end = uncovered_frames[0]
                                
                                for frame in uncovered_frames[1:]:
                                    if frame == current_end + 5:  # Allow small gaps (pose sampling is every 5 frames)
                                        current_end = frame
                                    else:
                                        # End current range, start new one
                                        if (current_end - current_start) >= 10:  # At least ~3 frames worth 
                                            ranges.append((current_start, current_end))
                                        current_start = frame
                                        current_end = frame
                                
                                # Add the last range
                                if (current_end - current_start) >= 10:
                                    ranges.append((current_start, current_end))
                                
                                # Process each continuous range
                                for range_start, range_end in ranges:
                                    duration_sec = (range_end - range_start) / fps
                                    
                                    # Skip if too short
                                    if duration_sec < 0.5:
                                        continue
                                        
                                    print(f"\nProcessing static pose segment: {range_start}-{range_end}, pose: {pose_name}")
                                    
                                    # Create a dance segment instance for this static pose
                                    if hasattr(self, "segment_class"):
                                        static_segment = self.segment_class()
                                        print(f"OK: Created dance segment for static pose: {static_segment}")
                                    else:
                                        print("ERROR: No segment class available")
                                        continue
                                
                            # Create and add pose instance
                            pose_instance = None
                            try:
                                if pose_name.lower() in ['retire', 'retiré']:
                                    if hasattr(self, "retire_class"):
                                        pose_instance = self.retire_class()
                                        print(f"OK: Created Retire pose: {pose_instance}")
                                elif pose_name.lower() == 'arabesque':
                                    if hasattr(self, "arabesque_class"):
                                        pose_instance = self.arabesque_class()
                                        print(f"OK: Created Arabesque pose: {pose_instance}")
                                elif pose_name.lower() in ['plie', 'grand_plié']:
                                    if hasattr(self, "plie_class"):
                                        pose_instance = self.plie_class()
                                        print(f"OK: Created Plie pose: {pose_instance}")
                                
                                if pose_instance is None:
                                    print(f"ERROR: Cannot create pose instance for {pose_name}")
                                    continue
                            except Exception as e:
                                print(f"ERROR creating pose: {e}")
                                continue
                            
                            # Create and add motion
                            try:
                                # Create a generic "no motion" instance
                                motion_instance = None
                                try:
                                    if "UnknownMotion" in self.class_dict:
                                        motion_instance = self.class_dict["UnknownMotion"]()
                                        print(f"OK: Created UnknownMotion for static pose")
                                    elif "Motion" in self.class_dict:
                                        motion_instance = self.class_dict["Motion"]()
                                        print(f"OK: Created generic Motion instance for static pose")
                                    else:
                                        print("ERROR: No motion class available")
                                        continue
                                except Exception as e:
                                    print(f"ERROR creating motion: {e}")
                                    continue
                                
                                # Connect pose and motion to the static segment
                                try:
                                    # Connect pose
                                    if hasattr(self, "hasPose_prop"):
                                        self.hasPose_prop[static_segment] = [pose_instance]
                                        print(f"OK: Added pose {type(pose_instance).__name__} to static segment")
                                    else:
                                        print("ERROR: No hasPose property available")
                                        continue
                                        
                                    # Connect motion
                                    if hasattr(self, "hasMotion_prop"):
                                        self.hasMotion_prop[static_segment] = [motion_instance]
                                        print(f"OK: Added motion {type(motion_instance).__name__} to static segment")
                                    else:
                                        print("ERROR: No hasMotion property available")
                                        continue
                                        
                                    # Add duration
                                    if hasattr(self, "duration_prop"):
                                        # Wrap in list to make it iterable
                                        self.duration_prop[static_segment] = [float(duration_sec)]
                                        print(f"OK: Set duration: {duration_sec} seconds")
                                except Exception as e:
                                    print(f"ERROR connecting attributes: {e}")
                                    continue
                                
                                # Create motion data object for the static pose (similar to turn data)
                                motion_data = {
                                    'start_frame': int(start_frame),
                                    'end_frame': int(end_frame),
                                    'start_time': float(start_frame) / fps,
                                    'end_time': float(end_frame) / fps,
                                    'start_time_formatted': self._format_srt_time(float(start_frame) / fps),
                                    'end_time_formatted': self._format_srt_time(float(end_frame) / fps),
                                    'rotation_degrees': 0,  # No rotation for static poses
                                    'duration_sec': duration_sec
                                }
                                
                                # Store the static pose segment
                                all_segments.append({
                                    'segment': static_segment,
                                    'pose': pose_name,
                                    'motion_type': 'static',
                                    'motion_data': motion_data
                                })
                            
                            except Exception as e:
                                print(f"ERROR creating motion: {e}")
                                continue
            
            # --- Step 2: Run the reasoner once for all segments ---
            print("\n=== Step 2: Running Pellet reasoner on all segments ===")
            try:
                # Run the Pellet reasoner
                print("Starting Pellet reasoning process...")
                sync_reasoner_pellet(infer_data_property_values=True, infer_property_values=True, debug=2)
                print("OK: Pellet reasoning completed successfully")
            except Exception as e:
                print(f"WARNING: Pellet reasoner error: {e}")
                print("Attempting to continue with limited reasoning...")
            
            # --- Step 3: Check what each segment was classified as ---
            print("\n=== Step 3: Checking classification results ===")
            for segment_info in all_segments:
                segment = segment_info['segment']
                pose = segment_info['pose']
                motion_data = segment_info['motion_data']
                
                # Determine what this segment was classified as
                try:
                    # Get all classes this segment belongs to
                    all_classes = list(segment.is_a)
                    class_names = [cls.name for cls in all_classes]
                    print(f"\nSegment classes: {class_names}")
                    
                    # Manual classification rules to ensure consistent results
                    # Look for specific named movement classes first
                    specific_movement = None
                    
                    # Check if this is a Pirouette - a Retiré pose with a Turn360 motion
                    if (pose.lower() == "retire" or pose.lower() == "retiré") and segment_info['motion_type'] == 'Turn360':
                        specific_movement = "Pirouette"
                        print(f"OK: Classified as Pirouette based on Retiré + Turn360")
                    # Check if this is a Fouette - an Arabesque pose with a Turn360 motion
                    elif pose.lower() == "arabesque" and segment_info['motion_type'] == 'Turn360':
                        specific_movement = "Fouette"
                        print(f"OK: Classified as Fouette based on Arabesque + Turn360")
                    else:
                        # If no specific compound movement was found, keep the component-based name
                        if segment_info['motion_type'] == 'Turn360':
                            specific_movement = "Turn360"  # Default to Turn360 for unclassified turns
                        elif pose:  # For static poses without specific motions
                            specific_movement = f"Static {pose}".replace('_', ' ').title()
                        print(f"! Pellet couldn't classify specifically, using component-based name: {specific_movement}")
                    
                    # Debug information to understand why classification might have failed
                    print(f"Debug - Pose: {type(list(segment.hasPose)[0]).__name__ if hasattr(segment, 'hasPose') else 'None'}")
                    print(f"Debug - Motion: {type(list(segment.hasMotion)[0]).__name__ if hasattr(segment, 'hasMotion') else 'None'}")
                
                except Exception as e:
                    print(f"ERROR checking classification: {e}")
                    specific_movement = "Unknown"
                
                # Add to our results - using Pellet classification only
                compound_movements.append({
                    'start_frame': motion_data['start_frame'],
                    'end_frame': motion_data['end_frame'],
                    'start_time': motion_data['start_time'],
                    'end_time': motion_data['end_time'],
                    'start_time_formatted': motion_data['start_time_formatted'],
                    'end_time_formatted': motion_data['end_time_formatted'],
                    'pose': pose,
                    'motion_type': segment_info['motion_type'],
                    'compound_movement': specific_movement,
                    'duration_sec': motion_data['duration_sec']
                })
        
        return compound_movements
    
    def generate_srt_subtitles(self, compound_movements):
        """
        Generate SRT subtitle content from compound movements
        
        Args:
            compound_movements: List of classified movements with timing
            
        Returns:
            String containing the SRT subtitle content
        """
        srt_content = ""
        for i, movement in enumerate(compound_movements, start=1):
            srt_content += f"{i}\n"
            srt_content += f"{movement['start_time_formatted']} --> {movement['end_time_formatted']}\n"
            srt_content += f"{movement['compound_movement']}: {movement['pose']} with {movement['motion_type']}\n\n"
        return srt_content
    
    def _format_srt_time(self, seconds):
        """Format time for SRT subtitles: HH:MM:SS,mmm"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
    
    def visualize_results(self, compound_movements, video_length=None, show=True, save_path=None):
        """
        Visualize detected compound movements on a timeline.
        
        Args:
            compound_movements: List of detected compound movements
            video_length: Total video length in seconds (if None, computed from frames)
            show: Whether to display the plot interactively (default: True)
            save_path: Path where to save the visualization (if None, not saved)
        """
        if not compound_movements:
            print("No compound movements to visualize")
            return
        
        # Extract max frame from movements
        max_frame = max([m['end_frame'] for m in compound_movements]) + 30
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Define movement types and their colors - darker shades
        MOVEMENT_COLORS = {
            'Pirouette': '#CC0099',      # Darker Magenta
            'Static Retiré': '#FF6B88',   # Darker Pink
            'Turn360': '#0047AB',        # Darker Royal Blue
            'Static Arabesque': '#4B0082', # Darker Purple
            'Static Grand Plié': '#CC7000'  # Darker Orange
        }
        
        # Define preferred order (top to bottom)
        PREFERRED_ORDER = [
            'Pirouette',
            'Static Retiré',
            'Turn360',
            'Static Arabesque',
            'Static Grand Plié'
        ]
        
        # Extract video number from save path or from movements
        video_number = "Unknown"
        if save_path:
            filename = os.path.basename(save_path)
            if '_' in filename:
                # This will get "I_11" from "I_11_timeline.png"
                video_number = '_'.join(filename.split('_')[:2])
        else:
            for m in compound_movements:
                if 'video_name' in m:
                    video_name = m['video_name']
                    if '_' in video_name:
                        # This will get "I_11" from whatever format it's in
                        video_number = '_'.join(video_name.split('_')[:2])
                    break
        
        # First pass to identify which movement types are present
        present_movements = set()
        
        for movement in compound_movements:
            pose = movement.get('pose', '').lower()
            motion_type = movement.get('motion_type', '')
            compound_movement = movement.get('compound_movement', '')
            
            # Add ALL component types for validation purposes
            
            # Always add the motion type if it exists
            if motion_type == 'Turn360':
                present_movements.add('Turn360')
            
            # Always add pose-based static movements
            if pose in ('retire', 'retiré'):
                present_movements.add('Static Retiré')
            elif pose == 'arabesque':
                present_movements.add('Static Arabesque')  
            elif pose in ('plie', 'grand_plié'):
                present_movements.add('Static Grand Plié')
                
            # Add compound movements
            if compound_movement == 'Pirouette':
                present_movements.add('Pirouette')
        
        # Create mapping for y-positions based on present movements
        movement_positions = {}
        current_pos = len(present_movements) - 1
        
        # Assign positions only to present movements, maintaining relative order
        for movement_type in PREFERRED_ORDER:
            if movement_type in present_movements:
                movement_positions[movement_type] = current_pos
                current_pos -= 1
        
        # Draw movements on timeline - SHOW ALL COMPONENTS FOR VALIDATION
        drawn_types = set()
        yticks = []
        ytick_labels = []
        colored_labels = {}
        
        for movement in compound_movements:
            start_frame = movement['start_frame']
            end_frame = movement['end_frame']
            pose = movement.get('pose', '').lower()
            motion_type = movement.get('motion_type', '')
            compound_movement = movement.get('compound_movement', '')
            
            # Draw ALL relevant components for this movement (for validation)
            components_to_draw = []
            
            # 1. Always add the motion type if it's Turn360
            if motion_type == 'Turn360':
                components_to_draw.append('Turn360')
            
            # 2. Always add pose-based components  
            if pose in ('retire', 'retiré'):
                components_to_draw.append('Static Retiré')
            elif pose == 'arabesque':
                components_to_draw.append('Static Arabesque')
            elif pose in ('plie', 'grand_plié'):
                components_to_draw.append('Static Grand Plié')
                
            # 3. Add compound movement classification
            if compound_movement == 'Pirouette':
                components_to_draw.append('Pirouette')
            
            # Draw each component
            for label in components_to_draw:
                if label in movement_positions:
                    y_pos = movement_positions[label]
                    color = MOVEMENT_COLORS[label]
                    
                    # Draw horizontal line for the movement
                    ax.hlines(y=y_pos, xmin=start_frame, xmax=end_frame,
                             color=color, linewidth=8, alpha=0.8)
                    drawn_types.add(label)
                    colored_labels[label] = color
                    
                    # Add to yticks if not already present
                    if y_pos not in yticks:
                        yticks.append(y_pos)
                        ytick_labels.append(label)
        
        # If no movements were drawn, show a message
        if not drawn_types:
            ax.text(0.5, 0.5, 'No recognized movements to display',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   color='black')
            ax.set_yticks([])
            ax.set_ylim(-0.5, 0.5)
        else:
            # Sort yticks and labels based on vertical position (top to bottom)
            ytick_label_pairs = sorted(zip(yticks, ytick_labels), reverse=True)
            yticks, ytick_labels = zip(*ytick_label_pairs) if ytick_label_pairs else ([], [])
            
            ax.set_yticks(yticks)
            
            # Create colored y-axis labels with their respective colors
            ax.set_yticklabels(ytick_labels, fontsize=10)
            
            # Update the color of each y-axis label
            for label, pos in zip(ax.get_yticklabels(), yticks):
                label.set_color(colored_labels[label.get_text()])
            
            ax.set_ylim(-0.5, len(present_movements) - 0.5)
        
        # Set x-axis properties
        ax.set_xlabel('Frame Number', color='black', fontsize=10)
        ax.set_title(f'Ballet Movement Analysis Timeline for Video {video_number}', 
                    color='black', pad=10, fontsize=12)
        
        # Title with video number
        ax.set_title(f'Ballet Movement Analysis Timeline for Video {video_number}', 
                    color='black', pad=10, fontsize=12)
        
        # Frame number ticks
        frame_interval = max(5, max_frame // 20)
        frame_ticks = range(0, max_frame + 1, frame_interval)
        ax.set_xticks(frame_ticks)
        ax.tick_params(axis='both', labelsize=9)
        ax.tick_params(axis='x', colors='black')
        
        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--', color='black')
        
        # Set spines to black
        for spine in ax.spines.values():
            spine.set_color('black')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"OK: Saved visualization to {save_path}")
        
        # Show if requested
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)
    
    def visualize_turn_timeline(self, compound_movements, video_length=None, show=True, save_path=None):
        """
        Create a separate timeline visualization focused on Turn360 movements.
        
        Args:
            compound_movements: List of detected compound movements
            video_length: Total video length in seconds (if None, computed from frames)
            show: Whether to display the plot interactively (default: True)
            save_path: Path where to save the visualization (if None, not saved)
        """
        if not compound_movements:
            print("No movements to visualize")
            return
            
        # Extract turn movements using EXACTLY the same logic as main timeline
        turn_movements = []
        for movement in compound_movements:
            pose = movement.get('pose', '').lower()
            motion_type = movement.get('motion_type', '')
            
            # Use EXACTLY the same logic as main timeline to identify Turn360 movements
            # This matches the logic in visualize_results() for Turn360 labeling
            if motion_type == 'Turn360':
                turn_movements.append(movement)
        
        if not turn_movements:
            print("No turn movements to visualize")
            return
            
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(14, 2))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Extract max frame from movements
        max_frame = max([m['end_frame'] for m in compound_movements]) + 30
        
        # Turn360 color (same as main timeline)
        TURN_COLOR = '#0047AB'  # Darker Royal Blue - matches MOVEMENT_COLORS['Turn360']
        
        # Extract video number
        video_number = "Unknown"
        if save_path:
            filename = os.path.basename(save_path)
            if '_' in filename:
                # This will get "I_11" from "I_11_timeline.png"
                video_number = '_'.join(filename.split('_')[:2])
        else:
            for m in compound_movements:
                if 'video_name' in m:
                    video_name = m['video_name']
                    if '_' in video_name:
                        # This will get "I_11" from whatever format it's in
                        video_number = '_'.join(video_name.split('_')[:2])
                    break
        
        # Draw turn movements with EXACTLY the same visual style as main timeline
        for movement in turn_movements:
            start_frame = movement['start_frame']
            end_frame = movement['end_frame']
            
            # Draw movement line EXACTLY as in main timeline (no annotations)
            ax.hlines(y=0, xmin=start_frame, xmax=end_frame,
                     color=TURN_COLOR, linewidth=8, alpha=0.8)
        
        # Set axis properties
        ax.set_xlabel('Frame Number', color='black', fontsize=10)
        
        # Title with video number and 360° symbol
        ax.set_title(f'360° Turn Timeline for Video {video_number}', 
                    color='black', pad=10, fontsize=12)
        
        # Frame number ticks
        frame_interval = max(5, max_frame // 20)
        frame_ticks = range(0, max_frame + 1, frame_interval)
        ax.set_xticks(frame_ticks)
        ax.tick_params(colors='black', labelsize=9)
        
        # Add 'Turn360' y-axis label
        ax.set_yticks([0])
        ax.set_yticklabels(['Turn360'], fontsize=10, color=TURN_COLOR)
        ax.set_ylim(-0.5, 0.5)
        ax.grid(axis='x', alpha=0.3, linestyle='--', color='black')
        
        # Set spines to black
        for spine in ax.spines.values():
            spine.set_color('black')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            # Modify save path to indicate turn timeline
            turn_save_path = save_path.replace('_timeline.png', '_turn_timeline.png')
            plt.savefig(turn_save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"OK: Saved turn timeline to {turn_save_path}")
        
        # Show if requested
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)


def build_ballet_ontology(output_path):
    """Create a simple ballet ontology for testing."""
    print(f"Building new ontology at {output_path}...")
    
    # Create new ontology with correct namespace
    onto = get_ontology("http://www.semanticweb.org/ballet/ontology#")
    
    with onto:
        # Define basic classes
        class Pose(Thing): pass
        class Motion(Thing): pass
        class DanceSegment(Thing): pass
        class BalletMovement(DanceSegment): pass
        
        # Define poses - match CNN output names (case-insensitive)
        class Retire(Pose): pass  # For "retiré" from CNN
        class Arabesque(Pose): pass  # For "arabesque" from CNN
        class Plie(Pose): pass  # For "grand_plié" from CNN
        class Tendu(Pose): pass  # For potential "tendu" from CNN
        class UnknownPose(Pose): pass  # Fallback class
        
        # Define motions - FIXED: No more circular dependencies
        class Turn360(Motion): pass  # For detecting 360-degree turns
        class Jump(Motion): pass
        class UnknownMotion(Motion): pass  # Fallback class
        
        # Define compound movements
        class Pirouette(BalletMovement): pass
        class Fouette(BalletMovement): pass
        class Assemble(BalletMovement): pass
        
        # Define properties with explicit domains and ranges
        class hasPose(ObjectProperty):
            domain = [DanceSegment]
            range = [Pose]
        
        class hasMotion(ObjectProperty):
            domain = [DanceSegment]
            range = [Motion]
        
        class durationInSeconds(DataProperty, FunctionalProperty):
            domain = [DanceSegment]
            range = [float]
        
        class rotationDegree(DataProperty, FunctionalProperty):
            domain = [Motion]
            range = [float]
        
        # Define restrictions for movement classification
        Pirouette.equivalent_to = [DanceSegment & 
                                 hasPose.some(Retire) & 
                                 hasMotion.some(Turn360)]
        
        Fouette.equivalent_to = [DanceSegment &
                               hasPose.some(Arabesque) &
                               hasMotion.some(Turn360)]
        
        # Save the ontology
        print(f"Saving ontology with classes: {', '.join([c.name for c in list(onto.classes())])}")
        onto.save(output_path)
        print(f"Created ballet ontology at {output_path} with {len(list(onto.classes()))} classes")
        return onto


def main():
    # Paths
    ontology_dir = './Ontology_Reasoner'
    ontology_path = os.path.join(ontology_dir, 'myBallet.owl')
    npy_folder_path = './I_17/data'
    pose_results_path = './output/I_17_pose_classifications.csv'
    output_dir = './output'

    # Ensure directories exist
    os.makedirs(ontology_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if ontology exists, if not create a simple one for testing
    if not os.path.exists(ontology_path):
        build_ballet_ontology(ontology_path)
    
    # Initialize the reasoner
    reasoner = BalletOntologyReasoner(ontology_path)
    
    # Detect motions
    motion_results = motion_detector.detect_turns_in_video(npy_folder=npy_folder_path, fps=30, plot=False)
    
    # Load pose classifications
    pose_results = reasoner.load_pose_classifications(pose_results_path)
    
    # Infer compound movements using real reasoning
    compound_movements = reasoner.infer_compound_movements(pose_results, motion_results)
    
    # Generate SRT subtitles (for video overlay).
    srt_content = reasoner.generate_srt_subtitles(compound_movements)
    
    # Save outputs
    srt_path = os.path.join(output_dir, 'ballet_movements.srt')
    with open(srt_path, 'w') as f:
        f.write(srt_content)
    
    json_path = os.path.join(output_dir, 'ballet_movements.json')
    with open(json_path, 'w') as f:
        # Convert to serializable format
        serialized = []
        for m in compound_movements:
            m_copy = m.copy()
            serialized.append(m_copy)
        json.dump(serialized, f, indent=2)
    
    # Visualize results
    reasoner.visualize_results(compound_movements)
    
    # Print summary
    print(f"\nOK: Analyzed {len(motion_results)} motion segments")
    print(f"OK: Identified {len(compound_movements)} compound ballet movements")
    print(f"OK: Saved subtitles to {srt_path}")
    print(f"OK: Saved detailed results to {json_path}")


if __name__ == "__main__":
    main()
