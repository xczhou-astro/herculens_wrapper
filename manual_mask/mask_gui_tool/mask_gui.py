import os
import sys
import subprocess
import argparse
import base64
import io
import json
import logging
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_file
import numpy as np
from PIL import Image
from astropy.io import fits

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Parse command-line arguments if launched directly
DEFAULT_BASE_DIR = Path(__file__).parent.parent.resolve()
TEMPLATE_DIR = Path(__file__).parent / "templates"

parser = argparse.ArgumentParser(description="Strong Lensing Mask & Crop GUI Server")
parser.add_argument("--file", "-f", "--fits_file", type=str, default=None, help="Direct FITS file path to mask")
parser.add_argument("--image_hdu", "-ih", type=str, default=None, help="HDU index or name for science image")
parser.add_argument("--noise_hdu", "-nh", type=str, default=None, help="HDU index or name for noise map")
parser.add_argument("--port", "-p", type=int, default=5000, help="Port to run GUI server on")
parser.add_argument("--use_rgb", action="store_true", default=False, help="Enable RGB visualization mode")
parser.add_argument("--rgb_file", type=str, default=None, help="Direct RGB image file path")
args, unknown = parser.parse_known_args()

def resolve_dir_path(path_str):
    """Resolve directory path, handling relative/absolute paths and '+' vs ' ' in query strings."""
    if not path_str:
        return None
    p = Path(path_str).expanduser().resolve()
    if p.exists():
        return p
    if " " in path_str:
        p_plus = Path(path_str.replace(" ", "+")).expanduser().resolve()
        if p_plus.exists():
            return p_plus
    return p if p.exists() else None

def resolve_fits_filepath(rel_path, base_dir=None):
    """Resolve direct FITS file path or relative candidate."""
    if not rel_path:
        return None
        
    p_input = Path(rel_path).expanduser()
    if p_input.is_absolute() and p_input.exists():
        return p_input.resolve()
        
    p_cwd = (Path.cwd() / p_input).resolve()
    if p_cwd.exists() and p_cwd.is_file():
        return p_cwd
        
    if base_dir:
        p_base = (base_dir / p_input).resolve()
        if p_base.exists() and p_base.is_file():
            return p_base
            
    if " " in rel_path:
        rel_plus = rel_path.replace(" ", "+")
        return resolve_fits_filepath(rel_plus, base_dir)
        
    return None

def parse_hdu_key(key):
    """Convert string key to int index if numeric, else return string."""
    if key is None or key == "":
        return None
    if isinstance(key, int):
        return key
    key_str = str(key).strip()
    if key_str.isdigit():
        return int(key_str)
    return key_str

def get_hdu_list_info(filepath):
    """Return list of 2D HDUs in FITS file with index, name, shape."""
    info = []
    try:
        with fits.open(filepath) as hdul:
            for idx, hdu in enumerate(hdul):
                if hdu.data is not None and hdu.data.ndim == 2:
                    extname = hdu.name if hdu.name else f"HDU_{idx}"
                    info.append({
                        "index": idx,
                        "name": extname,
                        "shape": list(hdu.data.shape),
                        "label": f"HDU {idx}: {extname} ({hdu.data.shape[1]}x{hdu.data.shape[0]})"
                    })
    except Exception as ex:
        logger.error(f"Error inspecting HDUs in {filepath}: {ex}")
    return info

def get_hdu_data(hdul, hdu_key, is_noise=False):
    """Get HDU data and header from hdul using index or name, or autodetect."""
    key = parse_hdu_key(hdu_key)
    target_hdu = None
    
    if key is not None:
        try:
            target_hdu = hdul[key]
        except (IndexError, KeyError):
            pass
            
    if target_hdu is None or target_hdu.data is None or target_hdu.data.ndim != 2:
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None and hdu.data.ndim == 2:
                name_upper = hdu.name.upper() if hdu.name else ""
                if is_noise:
                    if any(w in name_upper for w in ["ERR", "NOISE", "VAR", "WHT", "WEIGHT", "RMS"]):
                        target_hdu = hdu
                        break
                else:
                    if any(w in name_upper for w in ["SCI", "SCIENCE", "DATA", "IMAGE", "PRIMARY"]):
                        target_hdu = hdu
                        break
                        
        if target_hdu is None or target_hdu.data is None or target_hdu.data.ndim != 2:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    target_hdu = hdu
                    break
                    
    if target_hdu is not None and target_hdu.data is not None and target_hdu.data.ndim == 2:
        data = target_hdu.data
        name = target_hdu.name.upper() if target_hdu.name else ""
        if is_noise:
            if "WHT" in name or "WEIGHT" in name:
                weight = data.astype(float)
                rms = np.zeros_like(weight)
                valid = weight > 0
                rms[valid] = 1.0 / np.sqrt(weight[valid])
                rms[~valid] = np.nanmax(rms[valid]) if np.any(valid) else 1e-2
                data = rms
            elif "VAR" in name or "VARIANCE" in name:
                var = data.astype(float)
                rms = np.zeros_like(var)
                valid = var > 0
                rms[valid] = np.sqrt(var[valid])
                rms[~valid] = np.nanmax(rms[valid]) if np.any(valid) else 1e-2
                data = rms
        return data, target_hdu.header, target_hdu.name
    return None, None, None

TARGET_FILE = args.file
DEFAULT_IMAGE_HDU = args.image_hdu
DEFAULT_NOISE_HDU = args.noise_hdu
USE_RGB_MODE = args.use_rgb
RGB_FILE = args.rgb_file

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Disable caching for API responses
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/rgb_image")
def get_rgb_image():
    file_path = request.args.get("file")
    resolved = resolve_fits_filepath(file_path)
    if resolved and resolved.exists():
        return send_file(resolved)
    return "File not found", 404


@app.route("/api/select_file", methods=["GET", "POST"])
def select_file_dialog():
    """Open a native OS file dialog to select a FITS or RGB file."""
    mode = request.args.get("type", "fits")
    file_path = None
    if sys.platform == "darwin":
        try:
            if mode == "rgb":
                script = 'POSIX path of (choose file with prompt "Select RGB File:" of type {"png", "jpg", "jpeg", "fits", "fit", "fts"})'
            else:
                script = 'POSIX path of (choose file with prompt "Select FITS File:" of type {"fits", "fit", "fts"})'
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                file_path = res.stdout.strip()
        except Exception as ex:
            logger.error(f"osascript error: {ex}")

    if not file_path:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            if mode == "rgb":
                filetypes = [("Image/FITS files", "*.png *.jpg *.jpeg *.fits *.fit *.fts"), ("All files", "*.*")]
                title = "Select RGB File"
            else:
                filetypes = [("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")]
                title = "Select FITS File"
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes
            )
            root.destroy()
        except Exception as e:
            logger.error(f"Tkinter file dialog error: {e}")

    if file_path:
        return jsonify({"file": file_path, "success": True})
    return jsonify({"file": None, "success": False, "message": "No file selected"})

@app.route("/api/files", methods=["GET"])
def get_files():
    """Return initial target file if provided via CLI."""
    global TARGET_FILE, RGB_FILE
    req_file = request.args.get("file")
    if req_file:
        TARGET_FILE = req_file
        
    req_rgb = request.args.get("rgb_file")
    if req_rgb:
        RGB_FILE = req_rgb
        
    return jsonify({
        "target_file": str(TARGET_FILE) if TARGET_FILE else None,
        "rgb_file": str(RGB_FILE) if RGB_FILE else None,
        "use_rgb": USE_RGB_MODE,
        "default_image_hdu": DEFAULT_IMAGE_HDU,
        "default_noise_hdu": DEFAULT_NOISE_HDU
    })

@app.route("/api/load", methods=["GET"])
def load_file():
    """Load a FITS file and return its raw float data for client-side stretching."""
    rel_path = request.args.get("file")
    image_hdu_req = request.args.get("image_hdu") or DEFAULT_IMAGE_HDU
    noise_hdu_req = request.args.get("noise_hdu") or DEFAULT_NOISE_HDU
    
    if not rel_path:
        return jsonify({"message": "Missing file parameter"}), 400
        
    filepath = resolve_fits_filepath(rel_path)

    if filepath is None or not filepath.exists():
        return jsonify({"message": f"File not found: {rel_path}"}), 404
        
    try:
        logger.info(f"Loading FITS file: {filepath}")
        hdus_info = get_hdu_list_info(filepath)
        
        with fits.open(filepath) as hdul:
            data, header, extname = get_hdu_data(hdul, image_hdu_req, is_noise=False)
            
            if data is None:
                return jsonify({"message": "No 2D image data found in FITS file"}), 400
                
            height, width = data.shape
            
            flat_data = data.flatten().astype(float)
            finite_mask = np.isfinite(flat_data)
            finite_data = flat_data[finite_mask]
            
            if len(finite_data) > 0:
                min_val = float(finite_data.min())
                max_val = float(finite_data.max())
                flat_data[~finite_mask] = min_val
            else:
                min_val = 0.0
                max_val = 1.0
                flat_data = np.zeros_like(flat_data)
                
            flipped_data = np.flipud(flat_data.reshape(height, width)).flatten()
            
            # Extract noise map from external file or HDU for SNR calculation
            parent_dir = filepath.parent
            rms_file = None
            if noise_hdu_req is None:
                noise_names = ["noise_map.fits", "noise.fits", "rms_map.fits", "rms.fits", "error.fits"]
                if parent_dir.exists():
                    for p in parent_dir.iterdir():
                        if p.is_file() and p.name.lower() in noise_names:
                            rms_file = p
                            break
                if rms_file is None and "_sci" in filepath.name:
                    err_path = parent_dir / filepath.name.replace("_sci", "_err")
                    if err_path.exists():
                        rms_file = err_path
                    else:
                        wht_path = parent_dir / filepath.name.replace("_sci", "_wht")
                        if wht_path.exists():
                            rms_file = wht_path

            noise_data = None
            if rms_file and rms_file.exists():
                try:
                    with fits.open(rms_file) as hdul_err:
                        err_data, _, _ = get_hdu_data(hdul_err, None, is_noise=True)
                        if err_data is not None and err_data.shape == data.shape:
                            noise_data = err_data
                except Exception as ex:
                    logger.error(f"Error loading external noise file {rms_file}: {ex}")
                    
            if noise_data is None:
                err_data, _, _ = get_hdu_data(hdul, noise_hdu_req, is_noise=True)
                if err_data is not None and err_data.shape == data.shape:
                    noise_data = err_data
                    
            if noise_data is None:
                non_nan = data[np.isfinite(data)]
                bg_rms = float(np.nanstd(non_nan)) if len(non_nan) > 0 else 1e-2
                noise_data = np.full_like(data, bg_rms)
                
            flat_noise = noise_data.flatten().astype(float)
            valid_noise = flat_noise[np.isfinite(flat_noise) & (flat_noise > 0)]
            fallback_noise = float(np.median(valid_noise)) if len(valid_noise) > 0 else 1e-2
            flat_noise[~np.isfinite(flat_noise) | (flat_noise <= 0)] = fallback_noise
            flipped_noise = np.flipud(flat_noise.reshape(height, width)).flatten()
            
            filter_val = header.get("FILTER") or header.get("FILTERS") or header.get("BAND")
            instrument = header.get("INSTRUME")
            telescope = header.get("TELESCOP")
            
            return jsonify({
                "filepath": str(filepath),
                "parent_dir": str(filepath.parent / filepath.stem),
                "width": width,
                "height": height,
                "raw_data": flipped_data.tolist(),
                "raw_noise_data": flipped_noise.tolist(),
                "min_val": min_val,
                "max_val": max_val,
                "filter": str(filter_val) if filter_val else None,
                "instrument": str(instrument) if instrument else None,
                "telescope": str(telescope) if telescope else None,
                "hdus": hdus_info,
                "selected_image_hdu": extname
            })
            
    except Exception as e:
        logger.error(f"Error loading file {rel_path}: {e}", exc_info=True)
        return jsonify({"message": f"Error loading FITS: {str(e)}"}), 500

@app.route("/api/save", methods=["POST"])
def save_crop_and_masks():
    """Crop science data, crop error map, and save masks as FITS files."""
    try:
        req_data = request.json
        rel_path = req_data.get("file")
        center_x = float(req_data.get("center_x"))
        center_y = float(req_data.get("center_y"))
        box_size = int(req_data.get("box_size"))
        output_folder = req_data.get("output_folder", "").strip()
        image_hdu_req = req_data.get("image_hdu") or DEFAULT_IMAGE_HDU
        noise_hdu_req = req_data.get("noise_hdu") or DEFAULT_NOISE_HDU
        
        mask_layers = {
            "mask_1": req_data.get("mask_1"),
            "mask_2": req_data.get("mask_2"),
            "mask_out": req_data.get("mask_out")
        }
        
        if not rel_path:
            return jsonify({"message": "Missing file parameter"}), 400
            
        filepath = resolve_fits_filepath(rel_path)
            
        if filepath is None or not filepath.exists():
            return jsonify({"message": f"File not found: {rel_path}"}), 404
            
        # Determine output directory: default is filepath.parent / filepath.stem
        default_dir = filepath.parent / filepath.stem
        if output_folder:
            out_p = Path(output_folder).expanduser()
            if out_p.is_absolute():
                target_dir = out_p
            else:
                target_dir = (filepath.parent / out_p).resolve()
        else:
            target_dir = default_dir
            
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving crop and masks in directory: {target_dir}")
        
        # Load original science FITS file
        with fits.open(filepath) as hdul:
            data, header, img_extname = get_hdu_data(hdul, image_hdu_req, is_noise=False)
                    
            if data is None:
                return jsonify({"message": "No 2D image data found in FITS file"}), 400
                
            height, width = data.shape
            
            center_x_int = int(np.floor(center_x + 0.5))
            center_y_int = int(np.floor(center_y + 0.5))
            
            x0 = center_x_int - box_size // 2
            y0 = center_y_int - box_size // 2
            
            x0 = max(0, min(x0, width - box_size))
            y0 = max(0, min(y0, height - box_size))
            
            cropped_data = data[y0 : y0 + box_size, x0 : x0 + box_size]
            
            crop_header = header.copy()
            crop_header["NAXIS1"] = box_size
            crop_header["NAXIS2"] = box_size
            if "CRPIX1" in header:
                crop_header["CRPIX1"] = header["CRPIX1"] - x0
            if "CRPIX2" in header:
                crop_header["CRPIX2"] = header["CRPIX2"] - y0
                
            # Add header tags indicating source FITS file
            crop_header["ORIGFILE"] = (filepath.name, "Source FITS filename")
            crop_header["SRCFILE"] = (filepath.name, "Source FITS filename")
                
            cutout_path = target_dir / "Data_cutout.fits"
            fits.writeto(cutout_path, cropped_data.astype(np.float32), header=crop_header, overwrite=True)
            logger.info(f"Saved cutout to {cutout_path}")
            
            # Locate noise map: check external file first, or extract directly from HDU if noise_hdu specified/autodetected
            parent_dir = filepath.parent
            rms_file = None
            
            if noise_hdu_req is None:
                noise_names = ["noise_map.fits", "noise.fits", "rms_map.fits", "rms.fits", "error.fits"]
                if parent_dir.exists():
                    for p in parent_dir.iterdir():
                        if p.is_file() and p.name.lower() in noise_names:
                            rms_file = p
                            break
                            
                if rms_file is None and "_sci" in filepath.name:
                    err_path = parent_dir / filepath.name.replace("_sci", "_err")
                    if err_path.exists():
                        rms_file = err_path
                    else:
                        wht_path = parent_dir / filepath.name.replace("_sci", "_wht")
                        if wht_path.exists():
                            rms_file = wht_path

            rms_map_path = target_dir / "noise.fits"
            
            if rms_file and rms_file.exists():
                logger.info(f"Found external noise file: {rms_file}")
                with fits.open(rms_file) as hdul_err:
                    err_data, _, err_name = get_hdu_data(hdul_err, None, is_noise=True)
                    if err_data is not None:
                        if err_data.shape == data.shape:
                            cropped_err = err_data[y0 : y0 + box_size, x0 : x0 + box_size]
                        else:
                            cropped_err = err_data[y0 : y0 + box_size, x0 : x0 + box_size]
                            
                        fits.writeto(rms_map_path, cropped_err.astype(np.float32), header=crop_header, overwrite=True)
                        logger.info(f"Saved noise map to {rms_map_path}")
            else:
                # Extract noise directly from target FITS file HDU (specified noise_hdu or autodetected)
                err_data, _, noise_extname = get_hdu_data(hdul, noise_hdu_req, is_noise=True)
                if err_data is not None and err_data.shape == data.shape:
                    logger.info(f"Extracted noise map from HDU '{noise_extname}' in target FITS file.")
                    cropped_err = err_data[y0 : y0 + box_size, x0 : x0 + box_size]
                    fits.writeto(rms_map_path, cropped_err.astype(np.float32), header=crop_header, overwrite=True)
                    logger.info(f"Saved noise map to {rms_map_path}")
                else:
                    logger.info("No matching noise file or noise HDU found. Creating default mock noise.fits.")
                    non_nan = cropped_data[np.isfinite(cropped_data)]
                    bg_rms = float(np.nanstd(non_nan)) if len(non_nan) > 0 else 1e-2
                    mock_rms = np.full_like(cropped_data, bg_rms)
                    fits.writeto(rms_map_path, mock_rms.astype(np.float32), header=crop_header, overwrite=True)
                    logger.info(f"Saved mock noise map to {rms_map_path}")
            
        # Save mask layers
        for mask_name, mask_grid in mask_layers.items():
            if mask_grid is None:
                mask_grid = [[0 for _ in range(box_size)] for _ in range(box_size)]
                
            mask_array = np.array(mask_grid, dtype=np.uint8)
            fits_mask_data = np.flipud(mask_array)
            
            mask_path = target_dir / f"{mask_name}.fits"
            fits.writeto(mask_path, fits_mask_data, header=crop_header, overwrite=True)
            logger.info(f"Saved mask {mask_name} to {mask_path}")
            
        return jsonify({
            "message": "Outputs successfully saved",
            "cutout": str(cutout_path),
            "rms": str(rms_map_path),
            "noise": str(rms_map_path),
            "masks": [str(target_dir / f"{name}.fits") for name in mask_layers.keys()]
        })
        
    except Exception as e:
        logger.error(f"Error in saving crop and masks: {e}", exc_info=True)
        return jsonify({"message": f"Error saving outputs: {str(e)}"}), 500

@app.route("/api/load_masks", methods=["GET"])
def load_existing_masks():
    """Load existing masks if they exist in the output folder and match the expected box size."""
    try:
        output_folder = request.args.get("output_folder", "").strip()
        box_size = request.args.get("box_size")
        rel_path = request.args.get("file")

        if not box_size:
            return jsonify({"message": "Missing box_size parameter"}), 400
        box_size = int(box_size)
        
        target_dir = None
        if rel_path:
            filepath = resolve_fits_filepath(rel_path)
            if filepath:
                default_dir = filepath.parent / filepath.stem
                if output_folder:
                    out_p = Path(output_folder).expanduser()
                    target_dir = out_p if out_p.is_absolute() else (filepath.parent / out_p).resolve()
                else:
                    target_dir = default_dir
        if target_dir is None:
            if output_folder:
                out_p = Path(output_folder).expanduser()
                target_dir = out_p if out_p.is_absolute() else (Path.cwd() / out_p).resolve()
            else:
                target_dir = Path.cwd()
            
        masks = {}
        found_any = False
        
        for mask_name in ["mask_1", "mask_2", "mask_out"]:
            mask_path = target_dir / f"{mask_name}.fits"
            if mask_path.exists():
                logger.info(f"Loading existing mask: {mask_path}")
                try:
                    mask_data = fits.getdata(mask_path)
                    if mask_data.shape == (box_size, box_size):
                        canvas_mask = np.flipud(mask_data).astype(int)
                        masks[mask_name] = canvas_mask.tolist()
                        if np.any(canvas_mask):
                            found_any = True
                    else:
                        masks[mask_name] = None
                except Exception as ex:
                    logger.error(f"Error reading mask file {mask_path}: {ex}")
                    masks[mask_name] = None
            else:
                masks[mask_name] = None
                
        return jsonify({
            "found": found_any,
            "masks": masks
        })
        
    except Exception as e:
        logger.error(f"Error in loading existing masks: {e}", exc_info=True)
        return jsonify({"message": f"Error loading masks: {str(e)}"}), 500

def find_free_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

if __name__ == "__main__":
    port = args.port if args.port else 5000
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', port))
        s.close()
    except OSError:
        port = find_free_port()
        
    logger.info(f"Starting server on port {port}...")
    if TARGET_FILE:
        logger.info(f"Target FITS File: {TARGET_FILE}")
    if DEFAULT_IMAGE_HDU:
        logger.info(f"Default Image HDU: {DEFAULT_IMAGE_HDU}")
    if DEFAULT_NOISE_HDU:
        logger.info(f"Default Noise HDU: {DEFAULT_NOISE_HDU}")
    
    import webbrowser
    from threading import Timer
    
    def open_browser():
        webbrowser.open(f"http://127.0.0.1:{port}/")
        
    Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=port, debug=False)
