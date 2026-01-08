
import os
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add 'rendering' to path to load font loader
# In v4, we assume 'rendering' folder is present in pipeline_v4
sys.path.append(os.path.join(os.path.dirname(__file__), "rendering"))
try:
    from google_fonts_runtime_loader import get_font_path
except ImportError:
    print("Warning: Could not import google_fonts_runtime_loader. Fonts may fail.")
    # Fallback to importing from rendering package if sys path fails
    try:
        from rendering.google_fonts_runtime_loader import get_font_path
    except ImportError:
         pass

def composite_layers(run_dir, report_data):
    """
    Stack all cleaned layers to create the base 'clean' image.
    """
    layers_info = report_data["layer_cleaning"]["layers_processed"]
    layers_dir = Path(run_dir) / "layers"
    
    # Sort: 0 first
    layers_info.sort(key=lambda x: x["original_layer"])
    
    base_img = None
    
    print("\n[Compositing Layers]")
    for layer in layers_info:
        filename = layer["cleaned_layer"]
        path = layers_dir / filename
        
        if not path.exists():
            print(f"Warning: Layer {filename} not found.")
            continue
            
        print(f"  > Merging {filename}...")
        img = Image.open(path).convert("RGBA")
        
        if base_img is None:
            base_img = img
        else:
            if img.size != base_img.size:
                img = img.resize(base_img.size)
            base_img = Image.alpha_composite(base_img, img)
            
    return base_img

def draw_background_boxes(base_image, report_data, orig_w, orig_h, run_dir):
    """
    Composite extracted background box images for CTA regions.
    Uses the exact extracted pixels from the original layer, preserving gradients/textures.
    Only processes regions with background_box.detected = true.
    """
    img_w, img_h = base_image.size
    
    regions = report_data.get("text_detection", {}).get("regions", [])
    
    boxes_composited = 0
    print("\n[Compositing Extracted Background Boxes]")
    
    for region in regions:
        bg_box = region.get("background_box", {})
        
        if not bg_box.get("detected", False):
            continue
        
        # Check if we have an extracted image
        extracted_image_path = bg_box.get("extracted_image")
        layer_bbox = bg_box.get("layer_bbox", {})
        
        # Get region text for logging
        gemini = region.get("gemini_analysis", {})
        text_preview = gemini.get("text", "")[:20]
        
        if extracted_image_path and layer_bbox:
            # Load the extracted box image
            full_path = Path(run_dir) / extracted_image_path
            if full_path.exists():
                box_img = Image.open(full_path).convert("RGBA")
                
                # Get position from layer_bbox (already in layer/canvas coords)
                box_x = layer_bbox.get("x", 0)
                box_y = layer_bbox.get("y", 0)
                
                # Composite onto base image
                base_image.paste(box_img, (box_x, box_y), box_img)
                
                print(f"  > Composited '{text_preview}...' at ({box_x},{box_y}) "
                      f"[{box_img.size[0]}x{box_img.size[1]}]")
                boxes_composited += 1
            else:
                print(f"  ! Extracted image not found: {full_path}")
        else:
            # Fallback to solid color rectangle (shouldn't happen with new pipeline)
            print(f"  ! No extracted image for '{text_preview}...', skipping")
    
    print(f"  > Total boxes composited: {boxes_composited}")
    return base_image

def render_text_layer(base_image, report_data, scale_check=True):
    """
    Render text from JSON onto the base image.
    Uses Line-Based Grouping to align fragmented words (e.g. "Register" "For" "Free")
    into a single line, while strictly strictly preserving multi-line layouts (headlines).
    """
    draw = ImageDraw.Draw(base_image)
    
    # Dynamic Scale Calculation
    img_w, img_h = base_image.size
    
    # 1. Get original size from report (most robust)
    orig_w, orig_h = 1080, 1920 # Fallback
    
    if "original_size" in report_data:
        orig_w = report_data["original_size"]["width"]
        orig_h = report_data["original_size"]["height"]
        print(f"  > Using dimensions from report: {orig_w}x{orig_h}")
    else:
        # Fallback to file reading if not in report (legacy)
        input_filename = report_data.get("input_image", "")
        
        if input_filename:
            # Assumption: Input images are in "image/" folder relative to report or cwd
            # Try CWD/image first
            input_path = Path("image") / input_filename
            if input_path.exists():
                try:
                    with Image.open(input_path) as orig_img:
                        orig_w, orig_h = orig_img.size
                    print(f"  > Detected Original Source Size: {orig_w}x{orig_h}")
                except Exception as e:
                    print(f"Warning: Could not open source image {input_path}: {e}")
            else:
                 print(f"Warning: Source image not found at {input_path}. Using default.")
             
    ORIG_W, ORIG_H = orig_w, orig_h
    
    sx = img_w / ORIG_W
    sy = img_h / ORIG_H
    
    print(f"\n[Text Rendering]")
    print(f"  > Canvas Size: {img_w}x{img_h}")
    print(f"  > Scale Factors: x={sx:.3f}, y={sy:.3f} (Base: {ORIG_W}x{ORIG_H})")
    
    regions = report_data["text_detection"]["regions"]
    
    # 1. GROUP BY ROLE
    role_groups = {}
    for region in regions:
        gemini = region.get("gemini_analysis", {})
        if not gemini: continue
        
        # V4.13 FIX: Skip if text is residue on background (Qwen failed to layer)
        if region.get("layer_residue", False):
            print(f"  [SKIP] Region {region.get('id')} - Residue on Layer 0, not rendering.")
            continue
        
        role = gemini.get("role", "body")
        
        # V4.12 FIX: Hybrid Rendering for USP
        # If 'usp' was Extracted (detected=True), Render it.
        # If 'usp' was Protected (detected=False), Skip it (Preserved in BG).
        if role == "usp":
             bg_box = region.get("background_box", {})
             if not bg_box.get("detected", False):
                 continue # Preserved in BG, do not render ghost
        elif role not in ["heading", "subheading", "body", "cta"]:
            continue
            
        if role not in role_groups:
            role_groups[role] = []
        role_groups[role].append(region)
        
    # 2. CLUSTER INTO LINES & RENDER
    for role, group_regions in role_groups.items():
        if not group_regions: continue
        
        # Sort by Y first to help clustering
        group_regions.sort(key=lambda r: r["bbox"]["y"])
        
        lines = [] # List of lists (clusters)
        
        for r in group_regions:
            # Check overlap with existing lines
            placed = False
            r_y = r["bbox"]["y"]
            r_h = r["bbox"]["height"]
            r_center = r_y + r_h / 2
            
            for line in lines:
                # Vertical overlap check with the first element of the line
                # (Simple heuristic: overlapping centers or significant body overlap)
                l_r = line[0]
                l_y = l_r["bbox"]["y"]
                l_h = l_r["bbox"]["height"]
                l_center = l_y + l_h / 2
                
                # If vertical distance between centers is small relative to height
                # If vertical distance between centers is small relative to height
                avg_h = (r_h + l_h) / 2
                if abs(r_center - l_center) < (avg_h * 0.5):
                    # V4.9 FIX: Horizontal Gap Check
                    # If regions are horizontally far apart (e.g. > 1.5x height), do NOT merge.
                    # This preserves "Icon Label 1" ... "Icon Label 2" as separate render units.
                    
                    # V4.9.3: Bidirectional Horizontal Gap Check
                    # Handle cases where regions are processed out of X-order (due to Y-sorting)
                    line_min_x = min(lr["bbox"]["x"] for lr in line)
                    line_max_x = max(lr["bbox"]["x"] + lr["bbox"]["width"] for lr in line)
                    
                    r_min_x = r["bbox"]["x"]
                    r_max_x = r["bbox"]["x"] + r["bbox"]["width"]
                    
                    # Distance between intervals [line_min, line_max] and [r_min, r_max]
                    # If overlapping, distance is 0.
                    # If Disjoint: max(0, r_min - line_max, line_min - r_max)
                    gap = max(0, r_min_x - line_max_x, line_min_x - r_max_x)
                    
                    # Threshold: 0.6x Height (Aggressive split for columns)
                    threshold = avg_h * 0.6
                    
                    # print(f"      [Clustering Debug] Gap={gap:.1f}, Thresh={threshold:.1f}")
                    
                    if gap > threshold:
                        # Too far apart horizontally
                         continue 
                        
                    line.append(r)
                    placed = True
                    break
            
            if not placed:
                lines.append([r])
                
        # Render Each Line
        for line in lines:
            # Sort by X to form correct sentence
            line.sort(key=lambda r: r["bbox"]["x"])
            
            # Combine Text
            texts = [r["gemini_analysis"].get("text", "") for r in line]
            combined_text = " ".join(texts)
            if not combined_text.strip(): continue
            
            # Calculate Union Box
            min_x = min(r["bbox"]["x"] for r in line)
            min_y = min(r["bbox"]["y"] for r in line)
            max_x = max(r["bbox"]["x"] + r["bbox"]["width"] for r in line)
            max_y = max(r["bbox"]["y"] + r["bbox"]["height"] for r in line)
            
            orig_u_w = max_x - min_x
            orig_u_h = max_y - min_y
            
            # Scale
            x = min_x * sx
            y = min_y * sy
            w = orig_u_w * sx
            h = orig_u_h * sy
            
            # FIX B: Top-Left Anchor & Padding Strategy
            # Calculate pads based on SCALED height 'h'
            top_pad = h * 0.12
            bottom_pad = h * 0.08
            target_height = h - (top_pad + bottom_pad)
            
            print(f"  > Rendering Line '{combined_text}' at ({int(x)},{int(y)}) [{int(w)}x{int(h)}] ({role})")
            
            # Font Config (from first token)
            gemini = line[0]["gemini_analysis"]
            font_name = gemini.get("primary_font", "Roboto")
            weight = gemini.get("font_weight", 400)
            
            # Color Management
            color = gemini.get("text_color", "#000000")
            
            # BINARY SEARCH FONT SIZING
            def find_best_font_size(text, font_path, target_fit_height):
                low, high = 1, 2000
                best = low
                while low <= high:
                    mid = (low + high) // 2
                    try:
                        font = ImageFont.truetype(font_path, mid)
                        bbox = font.getbbox(text)
                        # bbox is (left, top, right, bottom)
                        text_height = bbox[3] - bbox[1]
                    except:
                        text_height = mid

                    if text_height <= target_fit_height:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                return best

            try:
                font_path = get_font_path(font_name, weight)
                
                # Sizing based on EFFECTIVE height (minus pads)
                font_size = find_best_font_size(combined_text, font_path, target_height)
                font = ImageFont.truetype(font_path, font_size)
                
                # Width Check (Clamp body, allow heading minor flow)
                left, top, right, bottom = font.getbbox(combined_text)
                text_w = right - left
                
                is_heading = role in ["heading", "hero_text"]
                width_limit = w * 1.1 if is_heading else w
                
                if text_w > width_limit:
                     scale_factor = width_limit / text_w
                     font_size = int(font_size * scale_factor)
                     font = ImageFont.truetype(font_path, font_size)

            except Exception as e:
                 print(f"    ! Font sizing failed: {e}")
                 font = ImageFont.load_default()
                 
            # ALIGNMENT FIX (Fix B)
            try:
                # Get precise bounding box relative to (0,0)
                text_bbox = font.getbbox(combined_text)
                # text_bbox = (left, top, right, bottom)
                
                # Horizontal Center (Standard)
                vis_text_width = text_bbox[2] - text_bbox[0]
                final_x = x + (w - vis_text_width) / 2 - text_bbox[0]
                
                # Vertical Alignment: TOP ANCHOR + CLAMP
                # Formula: visual_top aligns with (y + top_pad)
                final_y = (y + top_pad) - text_bbox[1]
                
                # ANTIGRAVITY HARD CLAMP:
                # "Never move rendered text DOWN relative to original bbox_top"
                # Visual Top is (final_y + text_bbox[1]).
                # Original Top Limit is 'y'.
                # Constraint: visual_top <= y
                
                visual_top = final_y + text_bbox[1]
                if visual_top > y:
                    # Clamp: shift final_y up so visual_top == y
                    final_y = y - text_bbox[1]
                
                draw.text((final_x, final_y), combined_text, fill=color, font=font)
                
            except Exception as e:
                print(f"    ! Alignment failed: {e}. Falling back to standard.")
                draw.text((x, y), combined_text, fill=color, font=font)
            
    return base_image

if __name__ == "__main__":
    # Hardcoded input for the specific run
    RUN_ID = "run_1767878379_layered"  
    BASE_DIR = Path("pipeline_outputs") / RUN_ID
    
    # Try to load report with boxes first, fallback to regular
    REPORT_WITH_BOXES = BASE_DIR / "pipeline_report_with_boxes.json"
    REPORT_PATH = BASE_DIR / "pipeline_report.json"
    
    if REPORT_WITH_BOXES.exists():
        print(f"Loading report with box data: {REPORT_WITH_BOXES}")
        with open(REPORT_WITH_BOXES, "r") as f:
            report = json.load(f)
    elif REPORT_PATH.exists():
        print(f"Box report not found, using: {REPORT_PATH}")
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
    else:
        print(f"Report not found: {REPORT_PATH}")
        sys.exit(1)
    
    # Get original image dimensions for scaling
    input_filename = report.get("input_image", "")
    orig_w, orig_h = 1080, 1920  # Default
    if input_filename:
        input_path = Path("image") / input_filename
        if input_path.exists():
            with Image.open(input_path) as orig_img:
                orig_w, orig_h = orig_img.size
                print(f"Original image size: {orig_w}x{orig_h}")
        
    # 1. Composite layers
    final_img = composite_layers(BASE_DIR, report)
    
    # 2. Composite extracted background boxes (for CTAs) BEFORE text
    final_img = draw_background_boxes(final_img, report, orig_w, orig_h, BASE_DIR)
    
    # 3. Render Text on top
    final_img = render_text_layer(final_img, report)
    
    # 4. Save
    out_path = BASE_DIR / "final_composed.png"
    final_img.save(out_path)
    print(f"\nSuccess! Final image saved to: {out_path}")
