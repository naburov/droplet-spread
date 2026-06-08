"""
Create a GIF from visualization images in an experiment directory.
"""

import os
import sys
import glob
from PIL import Image
import re


def extract_step_number(filename):
    """Extract step number from filename like 'joint_plot_step_250.png'."""
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def create_gif_from_experiment(
    experiment_dir,
    output_filename='animation.gif',
    duration=200,
    max_frames=None,
):
    """Create GIF from joint_plot images in experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        output_filename: Output GIF filename
        duration: Frame duration in milliseconds
    """
    vis_dir = os.path.join(experiment_dir, 'visualization')
    
    if not os.path.exists(vis_dir):
        print(f"Error: Visualization directory not found: {vis_dir}")
        sys.exit(1)
    
    # Find all joint_plot images
    pattern = os.path.join(vis_dir, 'joint_plot_step_*.png')
    image_files = glob.glob(pattern)
    
    if not image_files:
        print(f"Error: No joint_plot images found in {vis_dir}")
        sys.exit(1)
    
    # Sort by step number
    image_files.sort(key=extract_step_number)

    if max_frames is not None and max_frames > 0 and len(image_files) > max_frames:
        n = len(image_files)
        picks = [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]
        image_files = [image_files[i] for i in picks]

    print(f"Found {len(image_files)} images")
    print(f"First: {os.path.basename(image_files[0])}")
    print(f"Last: {os.path.basename(image_files[-1])}")
    
    # Load images
    print("Loading images...")
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            images.append(img.copy())
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    
    if not images:
        print("Error: No images could be loaded")
        sys.exit(1)
    
    # Create GIF
    output_path = os.path.join(experiment_dir, output_filename)
    print(f"Creating GIF: {output_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration: {duration} ms per frame")
    
    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # Loop forever
    )
    
    print(f"GIF created: {output_path}")
    print(f"  Total size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create_gif_from_experiment.py <experiment_dir> [output_filename] [duration_ms]")
        print("Example: python create_gif_from_experiment.py experiment_20260127_013540 animation.gif 200")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else 'animation.gif'
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None

    create_gif_from_experiment(experiment_dir, output_filename, duration, max_frames=max_frames)
