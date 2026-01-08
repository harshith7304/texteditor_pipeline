"""
Combined Pipeline Stage 2 & 3 (V4)
==================================
Combines Box Detection and Text Rendering into a single execution flow.
Usage: python run_pipeline_v4.py <RUN_ID_OR_DIR>

Stages:
1. Box Detection (run_pipeline_box_detection_v4.py)
   - Detects background boxes for CTA regions
   - Updates pipeline_report.json -> pipeline_report_with_boxes.json

2. Text Rendering (run_pipeline_text_rendering_v4.py)
   - Composites cleaned layers
   - Composites extracted background boxes
   - Renders text with dynamic sizing and alignment
   - Saves final_composed.png
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

# -----------------------------------------------------------------------------
# SETUP PATHS
# -----------------------------------------------------------------------------
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Add 'rendering' to path for font loader (required by text rendering module)
sys.path.append(os.path.join(os.path.dirname(__file__), "rendering"))

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
try:
    from run_pipeline_box_detection_v4 import run_box_detection_pipeline
    from run_pipeline_text_rendering_v4 import (
        composite_layers, 
        draw_background_boxes, 
        render_text_layer
    )
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Ensure run_pipeline_box_detection_v4.py and run_pipeline_text_rendering_v4.py are in the same directory.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# MAIN PIPELINE LOGIC
# -----------------------------------------------------------------------------

def run_pipeline_v4(run_dir: str):
    """
    Executes the combined Stage 2 (Box Detection) and Stage 3 (Rendering) pipeline.
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    print(f"\n************************************************************")
    print(f"STARTING COMBINED PIPELINE V4")
    print(f"Target: {run_dir}")
    print(f"************************************************************")

    # ---------------------------------------------------------
    # STAGE 2: BOX DETECTION
    # ---------------------------------------------------------
    print("\n>>> STAGE 2: Running Box Detection...")
    try:
        run_box_detection_pipeline(str(run_path))
    except Exception as e:
        print(f"!! Box Detection Failed: {e}")
        # We might continue if report exists, but usually this is fatal for the combined flow
        # However, rendering handles missing boxes gracefully.
    
    # ---------------------------------------------------------
    # STAGE 3: TEXT RENDERING
    # ---------------------------------------------------------
    print("\n>>> STAGE 3: Running Text Rendering...")
    
    # Load the updated report
    report_with_boxes = run_path / "pipeline_report_with_boxes.json"
    report_orig = run_path / "pipeline_report.json"
    
    report_file = report_with_boxes if report_with_boxes.exists() else report_orig
    
    if not report_file.exists():
        print(f"Error: No pipeline report found in {run_dir}")
        return
        
    print(f"Loading report: {report_file.name}")
    with open(report_file, "r") as f:
        report = json.load(f)
        
    # Get original image dimensions
    input_filename = report.get("input_image", "")
    orig_w, orig_h = 1080, 1920  # Fallback
    
    if input_filename:
        # Check standard locations
        possible_paths = [
            Path("image") / input_filename,
            Path(run_dir).parent.parent / "image" / input_filename, # ../../image
            Path("c:/Users/harsh/Downloads/zocket/product_pipeline/image") / input_filename # Absolute fallback based on known paths
        ]
        
        for p in possible_paths:
            if p.exists():
                try:
                    with Image.open(p) as orig_img:
                        orig_w, orig_h = orig_img.size
                    print(f"Original image size detected: {orig_w}x{orig_h} (from {p})")
                    break
                except:
                    continue
    
    try:
        # 1. Composite Layers
        final_img = composite_layers(str(run_path), report)
        if final_img is None:
            print("Error: Failed to composite layers.")
            return

        # 2. Composite Background Boxes (CTAs)
        final_img = draw_background_boxes(final_img, report, orig_w, orig_h, str(run_path))

        # 3. Render Text
        final_img = render_text_layer(final_img, report)

        # 4. Save Final Output
        out_path = run_path / "final_composed.png"
        final_img.save(out_path)
        print(f"\n>>> SUCCESS! Final combined image saved to:")
        print(f"{out_path}")
        
    except Exception as e:
        print(f"!! Text Rendering Failed: {e}")
        import traceback
        traceback.print_exc()

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Default to the most recent run ID edited by the user if no arg provided
    DEFAULT_RUN_ID = "run_1767874217_layered" 
    DEFAULT_BASE_DIR = Path("pipeline_outputs") / DEFAULT_RUN_ID
    
    target_dir = str(DEFAULT_BASE_DIR)
    
    # CLI Argument Override
    if len(sys.argv) > 1:
        # Check if arg is a full path or just a run ID
        arg_path = Path(sys.argv[1])
        if arg_path.exists():
            target_dir = str(arg_path)
        else:
            # Try appending to pipeline_outputs
            potential_dir = Path("pipeline_outputs") / sys.argv[1]
            if potential_dir.exists():
                target_dir = str(potential_dir)
            else:
                print(f"Warning: Provided path {sys.argv[1]} not found. Using default {DEFAULT_RUN_ID}")

    if not Path(target_dir).exists():
         print(f"Critical: Target directory {target_dir} does not exist.")
         sys.exit(1)

    run_pipeline_v4(target_dir)
