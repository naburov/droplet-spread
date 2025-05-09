import os
import argparse
import glob
from PIL import Image
import re

def natural_sort_key(s):
    """Sort strings with numbers in a natural way (e.g., 'step_10' comes before 'step_100')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_gif(input_dir, output_file='animation.gif', pattern='joint_plot_step_*.png', duration=200):
    """
    Create a GIF animation from PNG files in the specified directory.
    
    Args:
        input_dir (str): Directory containing PNG files
        output_file (str): Output GIF filename
        pattern (str): Glob pattern to match PNG files
        duration (int): Duration of each frame in milliseconds
    """
    # Get list of PNG files matching the pattern
    search_path = os.path.join(input_dir, pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"No files matching '{pattern}' found in '{input_dir}'")
        return False
    
    # Sort files naturally (so step_10 comes before step_100)
    files.sort(key=natural_sort_key)
    
    print(f"Found {len(files)} images to process")
    
    # Create GIF
    frames = []
    for file in files:
        try:
            img = Image.open(file)
            frames.append(img.copy())
            img.close()
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not frames:
        print("No valid images found")
        return False
    
    output_path = os.path.join(input_dir, output_file)
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )
    
    print(f"GIF created successfully: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create GIF from PNG files')
    parser.add_argument('input_dir', type=str, help='Directory containing PNG files')
    parser.add_argument('--output', type=str, default='animation.gif', help='Output GIF filename')
    parser.add_argument('--pattern', type=str, default='joint_plot_step_*.png', help='Glob pattern to match PNG files')
    parser.add_argument('--duration', type=int, default=200, help='Duration of each frame in milliseconds')
    
    args = parser.parse_args()
    
    create_gif(args.input_dir, args.output, args.pattern, args.duration)

if __name__ == "__main__":
    main()
