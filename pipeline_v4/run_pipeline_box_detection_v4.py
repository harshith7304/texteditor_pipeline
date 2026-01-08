"""
Background Box Detection Pipeline V4
=================================
Detects background containers (buttons, banners, boxes) behind text regions.

Approach:
1. Use CLEANED LAYERS (text removed) for edge detection
2. Find rectangular contours containing text center points
3. Store box metadata in JSON for rendering

Author: AI Assistant
Date: 2026-01-02
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
MIN_BOX_AREA = 100          # Minimum area for a valid box (px^2) - lowered for buttons
MAX_BOX_AREA_RATIO = 0.5    # Box can't be more than 50% of image area
CORNER_THRESHOLD = 0.03     # For approxPolyDP (higher = more lenient)
EDGE_DILATION = 5           # Dilate edges to close small gaps
DEBUG_SAVE_EDGES = True     # Save edge detection debug images

# -----------------------------------------------------------------------------
# CORE FUNCTIONS
# -----------------------------------------------------------------------------

# Import CraftTextDetector
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
try:
    from text_detector_craft_v4 import CraftTextDetector
except ImportError:
    print("Warning: Could not import CraftTextDetector. Local box cleaning may fail.")
    CraftTextDetector = None

def detect_boxes_in_layer(layer_path: str, text_regions: List[Dict], 
                          layer_scale: Tuple[float, float] = (1.0, 1.0)) -> List[Dict]:
    """
    Detect isolated background boxes in a layer using connected component analysis.
    
    Approach: Find pixel "islands" that are:
    1. Not connected to the image edges (isolated)
    2. Contain at least one text region center
    3. Are rectangular enough to be UI elements
    
    Args:
        layer_path: Path to the layer (original or cleaned)
        text_regions: List of text region dicts with 'bbox' keys
        layer_scale: (sx, sy) scale factors from original to layer dimensions
        
    Returns:
        List of detected box dicts with 'bbox', 'color', 'contains_regions'
    """
    # Load the layer
    layer = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)
    if layer is None:
        print(f"  ! Could not load layer: {layer_path}")
        return []
    
    h, w = layer.shape[:2]
    sx, sy = layer_scale
    
    # Create a binary mask of non-transparent, non-background pixels
    if len(layer.shape) == 3 and layer.shape[2] == 4:  # RGBA
        # Use alpha channel - pixels with alpha > threshold are "content"
        alpha = layer[:, :, 3]
        content_mask = (alpha > 20).astype(np.uint8) * 255
    else:
        # For RGB, create mask based on non-white/non-black pixels
        gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        # Assume very light (>250) or very dark (<5) are background
        content_mask = ((gray > 5) & (gray < 250)).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(content_mask, connectivity=8)
    
    print(f"    [DEBUG] Found {num_labels - 1} connected components (excluding background)")
    
    detected_boxes = []
    image_area = w * h
    
    for label_id in range(1, num_labels):  # Skip 0 (background)
        # Get component stats
        x, y, bw, bh, area = stats[label_id]
        
        # Filter by area
        if area < MIN_BOX_AREA:
            continue
        if area > image_area * MAX_BOX_AREA_RATIO:
            print(f"    [DEBUG] Component {label_id}: area {area} too large")
            continue
        
        # Check if component touches any edge (not isolated)
        touches_edge = (x == 0 or y == 0 or 
                        x + bw >= w - 1 or y + bh >= h - 1)
        
        if touches_edge:
            print(f"    [DEBUG] Component {label_id}: touches edge at ({x},{y}) size {bw}x{bh}")
            continue  # Skip non-isolated components
        
        # Check rectangularity (area vs bounding box area)
        bbox_area = bw * bh
        rectangularity = area / bbox_area if bbox_area > 0 else 0
        
        # V4.9.5 FIX: Relaxed Rectangularity back to 0.3 to catch stickers
        # (Bottle is now excluded by Role-Based Filter)
        if rectangularity < 0.3:
            print(f"    [DEBUG] Component {label_id}: low rectangularity={rectangularity:.2f}")
            continue
            
        print(f"    [DEBUG] Component {label_id} PASSED filters: area={area}, rect={rectangularity:.2f}")
        
        # Check which text regions this component contains
        contained_regions = []
        full_text_roles = []

        
        for region in text_regions:
            # FILTER: Include most text roles for box selection
            gemini = region.get("gemini_analysis", {})
            role = gemini.get("role", "body")
            
            # Scale text bbox to layer coordinates
            tx = region["bbox"]["x"] * sx
            ty = region["bbox"]["y"] * sy
            tw = region["bbox"]["width"] * sx
            th = region["bbox"]["height"] * sy
            
            # Check for ANY overlap between text bbox and component bbox
            x_overlap = max(0, min(x + bw, tx + tw) - max(x, tx))
            y_overlap = max(0, min(y + bh, ty + th) - max(y, ty))
            
            if x_overlap > 0 and y_overlap > 0:
                # Overlap logic...
                text_cx = max(x, min(x + bw - 1, tx + tw / 2))
                text_cy = max(y, min(y + bh - 1, ty + th / 2))
                cx_int, cy_int = int(text_cx), int(text_cy)
                cx_int = max(0, min(w - 1, cx_int))
                cy_int = max(0, min(h - 1, cy_int))
                
                overlap_area = x_overlap * y_overlap
                text_area = tw * th
                overlap_ratio = overlap_area / text_area if text_area > 0 else 0
                
                if labels[cy_int, cx_int] == label_id or overlap_ratio > 0.3:
                    contained_regions.append(region["id"])
                    full_text_roles.append(role)
                    # print(f"    [DEBUG] Component {label_id} contains region {region['id']} ({role})")

        # Only keep components that contain at least one text region
        if not contained_regions:
            print(f"    [DEBUG] Component {label_id}: no text regions contained")
            continue

        # V4.9.5 FIX: Role-Based Exclusion (Primary Fix)
        # If the box ONLY contains protected/preserved roles, it IS the object, not a box.
        # V4.12 UPDATE: Removed 'usp' from strict protected list (handled conditionally below)
        PROTECTED_ROLES = ["product_text", "logo", "icon", "label", "ui_element"]
        
        # Count how many contained regions are protected
        protected_count = sum(1 for r in full_text_roles if r in PROTECTED_ROLES)
        
        if protected_count == len(full_text_roles) and len(full_text_roles) > 0:
             print(f"    [FILTER] Dropping component {label_id} - Contains ONLY protected roles: {full_text_roles}")
             continue
             
        # If mixed, be careful. If majority is protected, skip.
        if protected_count > 0 and (protected_count / len(full_text_roles)) > 0.5:
              print(f"    [FILTER] Dropping component {label_id} - Majority protected roles: {full_text_roles}")
              continue

        # V4.12 FIX: USP Logic ("Solitary USP Only")
        # User Rule: "if its only usp then remove usp and extract bbox else if other text also present... protect"
        if "usp" in full_text_roles:
            # Check if there are ANY non-usp roles
            other_roles = [r for r in full_text_roles if r != "usp"]
            if other_roles:
                print(f"    [FILTER] Dropping component {label_id} - USP mixed with other text {other_roles} -> Protected")
                continue
            # Else: It is ONLY 'usp', so we allow it to proceed to extraction.


        # V4.9.4 FIX: Filter out Small Accessory Boxes (Icons/Bullets)
        # Logic: A valid "background box" (button, banner) should be LARGER than the text it contains.
        # If Box Area < Text Area, it's likely a small icon *next* to the text, strictly not a container.
        
        total_text_area = 0
        for rid in contained_regions:
            # Find region object
            r_obj = next((r for r in text_regions if r["id"] == rid), None)
            if r_obj:
                # Scale text area to layer coords
                tw = r_obj["bbox"]["width"] * sx
                th = r_obj["bbox"]["height"] * sy
                total_text_area += (tw * th)
        
        # Threshold: Box must be at least 60% of text area size?
        # Actually, for a button, Box > Text (Ratio > 1.0).
        # For a tight highlight, Box ~ Text (Ratio ~ 1.0).
        # For an icon bullet, Box << Text (Ratio < 0.2).
        # Safe threshold: 0.5. If box is half the size of the text, it's definitively NOT a container.
        
        area_ratio = area / total_text_area if total_text_area > 0 else 0
        print(f"    [DEBUG] Ratio Box/Text: {area:.0f}/{total_text_area:.0f} = {area_ratio:.2f}")
        
        if area_ratio < 0.6:
            print(f"    [FILTER] Dropping component {label_id} (Ratio {area_ratio:.2f} < 0.6) - likely icon/bullet")
            continue
        if not contained_regions:
            continue
        
        # Sample the dominant color inside the component
        component_mask = (labels == label_id).astype(np.uint8)
        box_color = sample_dominant_color_masked(layer, component_mask)
        
        # Scale back to original coordinates
        detected_boxes.append({
            "bbox": {
                "x": int(x / sx),
                "y": int(y / sy),
                "width": int(bw / sx),
                "height": int(bh / sy)
            },
            # Pass RAW layer coordinates to avoid rounding errors during extraction
            "layer_bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(bw),
                "height": int(bh)
            },
            "color": box_color,
            "contains_regions": contained_regions,
            "area": int(area / (sx * sy)),
            "rectangularity": round(rectangularity, 2),
            "is_isolated": True
        })
        
        print(f"    [FOUND] Isolated box: ({int(x/sx)}, {int(y/sy)}) "
              f"[{int(bw/sx)}x{int(bh/sy)}] rect={rectangularity:.2f} "
              f"contains: {contained_regions}")
    
    # Sort by area (smaller boxes first)
    detected_boxes.sort(key=lambda b: b["area"])
    
    return detected_boxes


def sample_dominant_color_masked(image: np.ndarray, mask: np.ndarray) -> str:
    """
    Sample the dominant color from a masked region.
    Returns hex color string.
    """
    if image is None or mask is None:
        return "#000000"
    
    # Get pixels where mask is non-zero
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            pixels = image[mask > 0][:, :3]  # Get BGR, ignore alpha
        else:
            pixels = image[mask > 0]
    else:
        return "#808080"
    
    if len(pixels) == 0:
        return "#000000"
    
    # Use median for dominant color
    median_color = np.median(pixels, axis=0).astype(int)
    
    # Convert BGR to RGB hex
    r, g, b = int(median_color[2]), int(median_color[1]), int(median_color[0])
    return f"#{r:02X}{g:02X}{b:02X}"


def sample_dominant_color(crop: np.ndarray) -> str:
    """
    Sample the dominant color from a cropped region.
    Returns hex color string.
    """
    if crop is None or crop.size == 0:
        return "#000000"
    
    # Flatten to list of pixels
    if len(crop.shape) == 3:
        if crop.shape[2] == 4:  # RGBA
            # Filter out transparent pixels
            pixels = crop.reshape(-1, 4)
            opaque_mask = pixels[:, 3] > 128
            if not np.any(opaque_mask):
                return "#000000"
            pixels = pixels[opaque_mask][:, :3]
        else:
            pixels = crop.reshape(-1, 3)
    else:
        return "#808080"  # Grayscale fallback
    
    if len(pixels) == 0:
        return "#000000"
    
    # Use k-means to find dominant color (k=1 for simplest case)
    # For speed, just use median
    median_color = np.median(pixels, axis=0).astype(int)
    
    # Convert BGR to RGB hex
    r, g, b = int(median_color[2]), int(median_color[1]), int(median_color[0])
    return f"#{r:02X}{g:02X}{b:02X}"


def assign_boxes_to_regions(text_regions: List[Dict], boxes: List[Dict]) -> List[Dict]:
    """
    Assign detected boxes to their corresponding text regions.
    Updates text_regions in place with 'background_box' field.
    """
    for region in text_regions:
        region_id = region["id"]
        
        # Find the smallest box that contains this region
        matching_boxes = [b for b in boxes if region_id in b["contains_regions"]]
        
        if matching_boxes:
            # Pick the smallest (most specific) box
            best_box = min(matching_boxes, key=lambda b: b["area"])
            region["background_box"] = {
                "detected": True,
                "bbox": best_box["bbox"],
                "color": best_box["color"],
                "is_isolated": best_box.get("is_isolated", False),
                "rectangularity": best_box.get("rectangularity", 0),
                "extracted_image": best_box.get("extracted_image"),
                "layer_bbox": best_box.get("layer_bbox")
            }
        else:
            region["background_box"] = {
                "detected": False
            }
    
    return text_regions


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def run_box_detection_pipeline(run_dir: str):
    """
    Run background box detection on a completed pipeline run.
    
    Args:
        run_dir: Path to the pipeline run directory (e.g., pipeline_outputs/run_XXXX_layered)
    """
    run_path = Path(run_dir)
    print(f"\n============================================================")
    print(f"BACKGROUND BOX DETECTION PIPELINE")
    print(f"============================================================")
    print(f"Run Directory: {run_dir}")
    
    report_path = run_path / "pipeline_report.json"
    layers_dir = run_path / "layers"
    
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        return
    
    with open(report_path, "r") as f:
        report = json.load(f)
        
    # Initialize global CRAFT detector for local box cleaning
    craft_detector = None
    if CraftTextDetector:
        # Check for model weights
        weights_path = Path(current_dir).parent / "CRAFT-pytorch" / "craft_mlt_25k.pth"
        if weights_path.exists():
            print(f"Loading CRAFT for local box cleaning from {weights_path}...")
            # Use CPU by default for safety, or check torch.cuda.is_available() inside class
            craft_detector = CraftTextDetector(model_path=str(weights_path)) 
        else:
            print(f"CRAFT weights not found at {weights_path}")
    
    text_regions = report.get("text_detection", {}).get("regions", [])
    print(f"Found {len(text_regions)} text regions")
    
    # Get original image size for scaling
    orig_size = report.get("original_size", {})
    if orig_size:
        orig_w = orig_size.get("width", 1024)
        orig_h = orig_size.get("height", 1536)
    else:
        # Infer from max text region coordinates
        max_x = max(r["bbox"]["x"] + r["bbox"]["width"] for r in text_regions) if text_regions else 1024
        max_y = max(r["bbox"]["y"] + r["bbox"]["height"] for r in text_regions) if text_regions else 1536
        # Round up to reasonable canvas size (add 10% margin)
        orig_w = int(max_x * 1.1)
        orig_h = int(max_y * 1.1)
        print(f"  [INFO] Inferred original size from text regions: {orig_w}x{orig_h}")
    
    # Process each cleaned layer
    all_boxes = []
    layer_info = report.get("layer_cleaning", {}).get("layers_processed", [])
    
    for layer_entry in layer_info:
        cleaned_layer_name = layer_entry.get("cleaned_layer")
        if not cleaned_layer_name:
            continue
        
        cleaned_layer_path = layers_dir / cleaned_layer_name
        if not cleaned_layer_path.exists():
            print(f"  ! Layer not found: {cleaned_layer_path}")
            continue
        
        # Get layer dimensions for scaling
        layer_img = cv2.imread(str(cleaned_layer_path), cv2.IMREAD_UNCHANGED)
        if layer_img is None:
            continue
        
        layer_h, layer_w = layer_img.shape[:2]
        sx = layer_w / orig_w
        sy = layer_h / orig_h
        
        print(f"\n[Layer] {cleaned_layer_name} ({layer_w}x{layer_h})")
        print(f"  Scale: {sx:.3f}x, {sy:.3f}y")
        
        # Detect boxes in this layer
        boxes = detect_boxes_in_layer(str(cleaned_layer_path), text_regions, (sx, sy))
        print(f"  -> Found {len(boxes)} candidate boxes")
        
        # EXTRACT AND CLEAN BOX ISOLATION
        original_layer_name = layer_entry.get("original_layer")
        if original_layer_name and boxes:
            original_layer_path = layers_dir / original_layer_name
            if original_layer_path.exists():
                from PIL import Image, ImageDraw
                orig_layer_img = Image.open(original_layer_path).convert("RGBA")
                
                # Load cleaned layer for erasure (making hole in background)
                cleaned_layer_pil = Image.open(cleaned_layer_path).convert("RGBA")
                modified_cleaned = False
                
                # Create extracted_boxes directory
                extracted_dir = layers_dir / "extracted_boxes"
                extracted_dir.mkdir(exist_ok=True)
                
                for idx, box in enumerate(boxes):
                    # USE RAW LAYER COORDINATES (Fixes rounding drift)
                    if "layer_bbox" in box:
                        bx = box["layer_bbox"]["x"]
                        by = box["layer_bbox"]["y"]
                        bw = box["layer_bbox"]["width"]
                        bh = box["layer_bbox"]["height"]
                    else:
                        # Fallback (Should not happen with updated component)
                        bx = int(box["bbox"]["x"] * sx)
                        by = int(box["bbox"]["y"] * sy)
                        bw = int(box["bbox"]["width"] * sx)
                        bh = int(box["bbox"]["height"] * sy)
                    
                    # Clamp to layer bounds
                    bx = max(0, bx)
                    by = max(0, by)
                    bw = min(bw, layer_w - bx)
                    bh = min(bh, layer_h - by)
                    
                    if bw > 0 and bh > 0:
                        # 1. Crop the box region from original layer
                        box_img = orig_layer_img.crop((bx, by, bx + bw, by + bh))
                        
                        # 2. LOCAL CLEANING: Run CRAFT on this crop
                        if craft_detector:
                            # Convert to numpy for CRAFT (RGB)
                            # PIL is RGB (if convert('RGB') used) or RGBA. 
                            # Convert to RGB array for CRAFT
                            box_rgb = box_img.convert("RGB")
                            box_cv = np.array(box_rgb)

                            # print(f"     [Cleaning] Running local CRAFT on box {idx}...")
                            local_report = craft_detector.detect_text(box_cv)
                            local_regions = local_report.get("regions", [])
                            
                            # Prepare drawing
                            draw_box = ImageDraw.Draw(box_img)
                            
                            # Parse detected color
                            try:
                                c_hex = box.get("color", "#000000")
                                r = int(c_hex[1:3], 16)
                                g = int(c_hex[3:5], 16)
                                b = int(c_hex[5:7], 16)
                                fill_color = (r, g, b, 255)
                            except:
                                fill_color = (0, 0, 0, 255)
                            
                            erased_count = 0
                            for l_reg in local_regions:
                                # Get polygon from local detection
                                poly = l_reg.get("polygon")
                                if poly:
                                    # Poly is already in box local coords
                                    # Convert list of lists to list of tuples
                                    draw_box.polygon([tuple(p) for p in poly], fill=fill_color)
                                    erased_count += 1
                                    
                            if erased_count > 0:
                                print(f"     [Cleaning] Erased {erased_count} local text regions from box using {c_hex}")
                            else:
                                print("     [Cleaning] No text found locally to erase.")

                        else:
                            # Fallback to erasing known global regions if CRAFT fails loading
                            print("     [Warning] CRAFT not available for local cleaning, skipping.")

                        # Save with unique name
                        region_ids = "_".join(map(str, box["contains_regions"]))
                        box_filename = f"box_region_{region_ids}.png"
                        box_path = extracted_dir / box_filename
                        box_img.save(box_path)
                        
                        # Store the path in box metadata
                        box["extracted_image"] = str(box_path.relative_to(run_path))
                        box["layer_bbox"] = {"x": bx, "y": by, "width": bw, "height": bh}
                        
                        print(f"     - Extracted box for regions {box['contains_regions']} -> {box_filename}")
                        
                        # 3. ERASE from Background (Cleaned Layer)
                        # We overwrite the region with 0 alpha
                        cleaned_layer_pil.paste((0, 0, 0, 0), (bx, by, bx + bw, by + bh))
                        modified_cleaned = True
                        print(f"     - Erased box region from background layer")
                
                if modified_cleaned:
                    cleaned_layer_pil.save(cleaned_layer_path)
                    print(f"     [UPDATE] Saved {cleaned_layer_name} with transparent holes.")

        for box in boxes:
            print(f"     - Box at ({box['bbox']['x']}, {box['bbox']['y']}) "
                  f"[{box['bbox']['width']}x{box['bbox']['height']}] "
                  f"contains regions: {box['contains_regions']}")
        
        all_boxes.extend(boxes)
    
    # Assign boxes to text regions
    print("\n[Assigning boxes to text regions]")
    text_regions = assign_boxes_to_regions(text_regions, all_boxes)
    
    # Count assignments
    assigned_count = sum(1 for r in text_regions if r.get("background_box", {}).get("detected"))
    print(f"  -> {assigned_count}/{len(text_regions)} regions have background boxes")
    
    # Update report
    report["text_detection"]["regions"] = text_regions
    report["box_detection"] = {
        "total_boxes_found": len(all_boxes),
        "regions_with_boxes": assigned_count
    }
    
    # Save updated report
    output_report_path = run_path / "pipeline_report_with_boxes.json"
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved: {output_report_path}")
    print("=" * 60)
    print("BOX DETECTION COMPLETE")
    print("=" * 60)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Hardcoded for testing
    RUN_DIR = "pipeline_outputs/run_1767874217_layered"
    run_box_detection_pipeline(RUN_DIR)
