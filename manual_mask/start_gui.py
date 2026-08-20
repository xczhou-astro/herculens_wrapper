import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Strong Lensing Mask & Crop GUI")
    parser.add_argument("--file", "-f", "--fits_file", type=str, default=None, help="Direct FITS file path to mask")
    parser.add_argument("--image_hdu", "-ih", type=str, default=None, help="HDU index or name for science image (e.g. 1, SCI)")
    parser.add_argument("--noise_hdu", "-nh", type=str, default=None, help="HDU index or name for noise map (e.g. 2, ERR)")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port to run GUI server on")
    parser.add_argument("--use_rgb", action="store_true", default=False, help="Enable RGB visualization mode")
    parser.add_argument("--rgb_file", type=str, default=None, help="Direct RGB image file path")
    
    args, unknown = parser.parse_known_args()
    
    tool_dir = Path(__file__).parent / "mask_gui_tool"
    script_path = tool_dir / "mask_gui.py"
    
    print(f"Launching Strong Lensing Mask & Crop GUI...")
    print(f"Script: {script_path}")
    if args.use_rgb:
        print("Mode: RGB Masking & Visualization")
        if args.rgb_file:
            print(f"Target RGB File: {args.rgb_file}")
        if args.file:
            print(f"Target FITS File: {args.file}")
    else:
        print("Mode: Standard FITS Masking")
        if args.file:
            print(f"Target FITS File: {args.file}")
            
    if args.image_hdu:
        print(f"Image HDU: {args.image_hdu}")
    if args.noise_hdu:
        print(f"Noise HDU: {args.noise_hdu}")
    print(f"Port: {args.port}")
    
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nGUI Server stopped by user.")
    except Exception as e:
        print(f"\nError running GUI Server: {e}")
