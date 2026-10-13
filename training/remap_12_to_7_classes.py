"""
================================================================================
HANDWASH DATASET: DIRECTORY REMAPPING (12 → 7 CLASSES)
================================================================================
300 videos total: 25 per step × 12 steps
Merging left/right variants into single classes
================================================================================
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

# ============================================================================
# CONFIGURATION - CORRECTED PATH!
# ============================================================================

INPUT_DIR = '/kaggle/input/datasets/realtimear/hand-wash-dataset/HandWashDataset/HandWashDataset'
OUTPUT_DIR = '/kaggle/working/handwash_7class_remapped'

# Directory name mapping: original → new class
DIRECTORY_MAPPING = {
    'Step_1': 1,
    'Step_2_Left': 2,
    'Step_2_Right': 2,     # MERGED with Step_2_Left
    'Step_3': 3,
    'Step_4_Left': 4,
    'Step_4_Right': 4,     # MERGED with Step_4_Left
    'Step_5_Left': 5,
    'Step_5_Right': 5,     # MERGED with Step_5_Left
    'Step_6_Left': 6,
    'Step_6_Right': 6,     # MERGED with Step_6_Left
    'Step_7_Left': 7,
    'Step_7_Right': 7,     # MERGED with Step_7_Left
}

print("="*80)
print("HANDWASH DATASET: DIRECTORY REMAPPING (12 → 7)")
print("="*80)
print(f"\n📂 Input: {INPUT_DIR}")
print(f"📂 Output: {OUTPUT_DIR}")
print(f"\n📊 Dataset: 300 videos total (25 per step)")

print(f"\n📋 Remapping plan:")
print("   Step_1 (25)                       → class_1 (25)")
print("   Step_2_Left (25) + Step_2_Right (25) → class_2 (50) MERGED")
print("   Step_3 (25)                       → class_3 (25)")
print("   Step_4_Left (25) + Step_4_Right (25) → class_4 (50) MERGED")
print("   Step_5_Left (25) + Step_5_Right (25) → class_5 (50) MERGED")
print("   Step_6_Left (25) + Step_6_Right (25) → class_6 (50) MERGED")
print("   Step_7_Left (25) + Step_7_Right (25) → class_7 (50) MERGED")

# Create output directories for 7 classes
for class_id in range(1, 8):
    Path(OUTPUT_DIR, f'class_{class_id}').mkdir(parents=True, exist_ok=True)

# ============================================================================
# SCAN INPUT DIRECTORIES
# ============================================================================

print(f"\n{'='*80}")
print("SCANNING INPUT DIRECTORIES")
print(f"{'='*80}\n")

input_path = Path(INPUT_DIR)
found_dirs = {}

for dir_name, target_class in DIRECTORY_MAPPING.items():
    dir_path = input_path / dir_name
    
    if dir_path.exists() and dir_path.is_dir():
        # Count video files
        video_files = list(dir_path.glob('*.mp4')) + \
                     list(dir_path.glob('*.avi')) + \
                     list(dir_path.glob('*.mov')) + \
                     list(dir_path.glob('*.MP4')) + \
                     list(dir_path.glob('*.AVI')) + \
                     list(dir_path.glob('*.MOV'))
        
        found_dirs[dir_name] = {
            'path': dir_path,
            'target_class': target_class,
            'file_count': len(video_files),
            'files': video_files
        }
        
        print(f"OK {dir_name:20s} -> class_{target_class}  ({len(video_files):3d} videos)")
    else:
        print(f"XX {dir_name:20s} -> NOT FOUND")

# ============================================================================
# COPY FILES TO NEW STRUCTURE
# ============================================================================

print(f"\n{'='*80}")
print("COPYING FILES TO NEW STRUCTURE")
print(f"{'='*80}\n")

stats = {
    'total_files': 0,
    'files_per_class': defaultdict(int),
    'videos_per_original_dir': {},
}

print("Processing:\n")

for dir_name, info in sorted(found_dirs.items()):
    target_class = info['target_class']
    files = info['files']
    
    print(f"   {dir_name:20s} -> class_{target_class}", end="  ")
    
    # Copy each file
    for file_path in files:
        # Create new filename to preserve origin and avoid conflicts
        new_filename = f"{dir_name}_{file_path.name}"
        dest_path = Path(OUTPUT_DIR) / f'class_{target_class}' / new_filename
        
        # Copy file
        shutil.copy2(file_path, dest_path)
        
        stats['total_files'] += 1
        stats['files_per_class'][target_class] += 1
    
    stats['videos_per_original_dir'][dir_name] = len(files)
    print(f"OK Copied {len(files)} videos")

# ============================================================================
# FINAL STATISTICS
# ============================================================================

print(f"\n{'='*80}")
print("REMAPPING COMPLETE!")
print(f"{'='*80}\n")

print(f"Summary:")
print(f"   Total videos processed: {stats['total_files']:,}\n")

print(f"Original distribution (12 directories):")
for dir_name in sorted(stats['videos_per_original_dir'].keys()):
    count = stats['videos_per_original_dir'][dir_name]
    target_class = DIRECTORY_MAPPING[dir_name]
    print(f"   {dir_name:20s} -> class_{target_class}  |  {count:3d} videos")

print(f"\nNew distribution (7 classes - MERGED):")
total_files = stats['total_files']
for class_id in range(1, 8):
    count = stats['files_per_class'][class_id]
    percentage = (count / total_files * 100) if total_files > 0 else 0
    
    # Indicate which classes are merged
    if class_id in [2, 4, 5, 6, 7]:
        merged_tag = "MERGED (left + right)"
    else:
        merged_tag = ""
    
    print(f"   class_{class_id}: {count:4d} videos ({percentage:5.2f}%)  {merged_tag}")

# Check for class imbalance
counts = [stats['files_per_class'][i] for i in range(1, 8) if stats['files_per_class'][i] > 0]
if counts:
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else 0
    
    print(f"\nClass balance:")
    print(f"   Largest class: {max_count} videos")
    print(f"   Smallest class: {min_count} videos")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2.0:
        print(f"   WARNING: Moderate imbalance - Consider data augmentation!")
    else:
        print(f"   OK: Well balanced dataset!")

# ============================================================================
# SAVE METADATA
# ============================================================================

metadata = {
    'input_directory': str(INPUT_DIR),
    'output_directory': str(OUTPUT_DIR),
    'total_videos': stats['total_files'],
    'original_classes': 12,
    'remapped_classes': 7,
    'directory_mapping': DIRECTORY_MAPPING,
    'videos_per_class': dict(stats['files_per_class']),
    'videos_per_original_dir': stats['videos_per_original_dir'],
    'class_descriptions': {
        1: 'Step 1',
        2: 'Step 2 (merged: left + right)',
        3: 'Step 3',
        4: 'Step 4 (merged: left + right)',
        5: 'Step 5 (merged: left + right)',
        6: 'Step 6 (merged: left + right)',
        7: 'Step 7 (merged: left + right)',
    }
}

metadata_path = Path(OUTPUT_DIR) / 'remapping_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved: {metadata_path}")

# ============================================================================
# VERIFY OUTPUT STRUCTURE
# ============================================================================

print(f"\n{'='*80}")
print("OUTPUT DIRECTORY STRUCTURE")
print(f"{'='*80}\n")

print(f"{OUTPUT_DIR}/")
for class_id in range(1, 8):
    class_dir = Path(OUTPUT_DIR) / f'class_{class_id}'
    video_count = len(list(class_dir.glob('*.mp4')))
    
    # Show sample filenames
    sample_files = list(class_dir.glob('*.mp4'))[:2]
    
    print(f"   |-- class_{class_id}/  ({video_count} videos)")
    for sample in sample_files:
        print(f"   |   |-- {sample.name}")
    if video_count > 2:
        print(f"   |   +-- ... ({video_count - 2} more)")

print(f"\n{'='*80}")
print("DATASET READY!")
print(f"{'='*80}\n")

print("Next steps:")
print("1. OK Class remapping complete (12 -> 7)")
print("2. OK Dataset is balanced (except class_1 and class_3 have 25 vs 50)")
print("3. OK Ready for frame extraction or direct video processing")
print("4. OK Ready for CNN+LSTM model training")

print(f"\nUse this path for your CNN+LSTM training:")
print(f"   DATA_DIR = '{OUTPUT_DIR}'")

print(f"\n{'='*80}\n")