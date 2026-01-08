"""
Layered Pipeline V4 with Qwen + CRAFT + Gemini
===========================================
1. CRAFT: Detect text regions
2. Gemini: Classify regions (Keep vs Remove)
3. Qwen: Split image into layers (Text, Logo, UI, Base)
4. Cleanup: Erase "Remove" text pixels from Qwen layers deterministically
"""

import os
import sys
import time
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "qwen_layered_runner")) # v4: File is now local

# Import stages (Updated for v4)
try:
    from text_detector_craft_v4 import CraftTextDetector
    from gemini_text_analysis_pro_v4 import analyze_text_crops_batch
    from run_qwen_layered_v4 import run_qwen_layered
except ImportError:
    # Fallback if running from root relative
    from pipeline_v4.text_detector_craft_v4 import CraftTextDetector
    from pipeline_v4.gemini_text_analysis_pro_v4 import analyze_text_crops_batch
    from pipeline_v4.run_qwen_layered_v4 import run_qwen_layered

load_dotenv()

def get_layer_scale(orig_w, orig_h, layer_w, layer_h):
    if orig_w == 0 or orig_h == 0: return 0, 0
    return layer_w / orig_w, layer_h / orig_h

def clean_layers(layer_paths, output_dir, global_craft_result, gemini_results, orig_size, original_img=None):
    """
    Refined Layer Cleaning Logic (V4.2 High-Res Gating):
    1.  **Global Mapping**: Use global CRAFT boxes.
    2.  **High-Res Pixel Gating**: 
        - Crop from **ORIGINAL IMAGE** (High Precision).
        - Run CRAFT on high-res crop (`merge_lines=False`).
        - Map detected polygons -> Layer Coordinates.
        - Erase on Layer.
    3.  **Layer 0 Protection**: SACRED.
    """
    cleaned_paths = []
    cleaning_report = []

    # Roles configuration
    # Roles configuration
    # V4.11 UPDATE: Added 'usp' to PRESERVE, removed valid 'hero_text'
    PRESERVE_ROLES = ["product_text", "logo", "ui_element", "label", "icon", "usp"]
    REMOVE_ROLES   = ["heading", "subheading", "body", "cta"]
    
    # Initialize detector for LOCAL PIXEL GATING
    # CRITICAL: merge_lines=False
    print("  [Init] Loading CRAFT for pixel gating (High-Res Mode)...")
    detector = CraftTextDetector(cuda=False, merge_lines=False, link_threshold=0.2, text_threshold=0.6, low_text=0.35)
    
    gemini_map = {str(res["region_id"]): res["analysis"] for res in gemini_results}
    orig_h, orig_w = orig_size[:2]
    
    # Ensure original image is available for high-res cropping
    if original_img is None:
        print("  [Warning] Original Image not provided! Falling back to low-res layer cropping (Quality degraded).")

    for layer_path in layer_paths:
        path = Path(layer_path)
        if not path.exists(): continue
            
        print(f"  > Processing layer: {path.name}")
        
        # RULE 1: Skip Layer 0
        if "layer_0" in path.name.lower():
            print("    [Info] Layer 0 is Background -> PROTECTED")
            cleaned_filename = path.stem + "_cleaned.png"
            cleaned_path = output_dir / cleaned_filename
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None: cv2.imwrite(str(cleaned_path), img)
            cleaned_paths.append(str(cleaned_path))
            
            cleaning_report.append({
                "original_layer": path.name,
                "cleaned_layer": cleaned_filename,
                "status": "preserved",
                "action": "none (background)",
                "regions_removed": 0
            })
            continue

        # Process Overlay Layers
        raw_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw_img is None: continue
        
        # Ensure RGBA
        if raw_img.shape[2] == 3: raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2BGRA)
        
        h, w = raw_img.shape[:2]
        sx, sy = get_layer_scale(orig_w, orig_h, w, h)
        print(f"    @ Scale: x={sx:.3f}, y={sy:.3f} (Resolution: {w}x{h})")

        erasure_mask = np.zeros((h, w), dtype=np.uint8)
        regions_removed = 0
        pixel_gating_stats = {"checked": 0, "verified_text": 0, "skipped_graphic": 0}
        
        # V4.8 FIX: Atomic Erasure (Erase-All, Render-Once)
        # Trust Global CRAFT Polygon completely. No local re-detection.
        
        for region in global_craft_result["text_regions"]:
            rid = str(region["id"])
            role = gemini_map.get(rid, {}).get("role", "body").lower().strip()
            
            # 1. Logic: Only remove "Removable" roles
            if role not in REMOVE_ROLES: continue
            
            # 2. Logic: Atomic Erasure of Global Polygon
            # Use POLYGON from Global Detection (High Precision)
            g_poly = np.array(region["polygon"], dtype=np.float32) # Shape (N, 2)
            
            # Scale to Layer Coordinates
            # (Global X * sx = Layer X)
            l_poly = g_poly.copy()
            l_poly[:, 0] *= sx
            l_poly[:, 1] *= sy
            l_poly = l_poly.astype(np.int32)
            
            # 3. Fill Mask
            cv2.fillPoly(erasure_mask, [l_poly], 255)
            regions_removed += 1

        # 4. Apply Erasure
        if regions_removed > 0:
            # Dilation: Moderate to catch anti-aliasing (Atomic means wipe it ALL)
            kernel = np.ones((2, 2), np.uint8) 
            erasure_mask = cv2.dilate(erasure_mask, kernel, iterations=1)
            raw_img[:, :, 3] = np.where(erasure_mask == 255, 0, raw_img[:, :, 3])
            
            log = f"atomic_erasure (removed {regions_removed} regions)"
            print(f"    -> {log}")
        else:
            log = "no_erasures"
            print(f"    -> No regions matched.")

        # Save
        cleaned_filename = path.stem + "_cleaned.png"
        cleaned_path = output_dir / cleaned_filename
        cv2.imwrite(str(cleaned_path), raw_img)
        cleaned_paths.append(str(cleaned_path))
        
        cleaning_report.append({
            "original_layer": path.name,
            "cleaned_layer": cleaned_filename,
            "status": "cleaned" if regions_removed > 0 else "preserved",
            "action": log,
            "regions_removed": regions_removed
        })

    return cleaned_paths, cleaning_report

def detect_layer0_residue(layer0_path, global_regions, gemini_map, orig_size):
    """
    V4.13: Detect if 'Removable' text is still visible on Layer 0 (Background).
    If Qwen failed to layer text properly, it stays on Layer 0.
    We detect it here and mark those regions to skip rendering.
    
    Returns: List of region IDs that have residue on Layer 0.
    """
    REMOVE_ROLES = ["heading", "subheading", "body", "cta", "usp"]
    
    print("\n[STEP 4.5] Detecting Layer 0 Residue...")
    
    # 1. Run CRAFT on Layer 0
    detector = CraftTextDetector(cuda=False, merge_lines=True, text_threshold=0.5)
    layer0_img = cv2.imread(str(layer0_path))
    if layer0_img is None:
        print("  [Warning] Could not load Layer 0 for residue check.")
        return []
    
    layer0_result = detector.detect(layer0_img)
    l0_regions = layer0_result.get("text_regions", [])
    print(f"  > CRAFT found {len(l0_regions)} text regions on Layer 0.")
    
    if not l0_regions:
        print("  > No text detected on Layer 0. All clear!")
        return []
    
    # 2. Get Layer Scale
    l_h, l_w = layer0_img.shape[:2]
    orig_h, orig_w = orig_size[:2]
    sx, sy = l_w / orig_w, l_h / orig_h
    
    residue_ids = []
    
    # 3. For each Global Region marked as Removable, check if it's on Layer 0
    for region in global_regions:
        rid = str(region["id"])
        role = gemini_map.get(rid, {}).get("role", "body").lower().strip()
        
        if role not in REMOVE_ROLES:
            continue  # Only check Removable roles
        
        # Scale global bbox center to layer coords
        g_bbox = region["bbox"]
        l_cx = (g_bbox["x"] + g_bbox["width"] / 2) * sx
        l_cy = (g_bbox["y"] + g_bbox["height"] / 2) * sy
        
        # Check if any Layer 0 detection overlaps this center
        for l0_region in l0_regions:
            l0_poly = np.array(l0_region["polygon"], dtype=np.int32)
            if cv2.pointPolygonTest(l0_poly, (l_cx, l_cy), False) >= 0:
                text_preview = gemini_map.get(rid, {}).get("text", "")[:20]
                print(f"  [Residue] Region {rid} ('{role}': '{text_preview}...') found on Layer 0!")
                residue_ids.append(rid)
                break
    
    if residue_ids:
        print(f"  > Total residue regions: {len(residue_ids)}")
    else:
        print("  > No removable text residue detected on Layer 0.")
    
    return residue_ids

def run_pipeline_layered(image_path_str: str, mock_layers_dir: str = None) -> str:
    """
    Run the full layered pipeline on an image.
    Args:
        image_path_str: Path to the input image
        mock_layers_dir: Optional path to pre-existing layers to skip Qwen cost
    Returns:
        str: Path to the run directory
    """
    image_path = Path(image_path_str)
    output_base = "pipeline_outputs"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Error: {image_path} not found")

    # Setup run dir
    run_id = int(time.time())
    run_dir = Path(output_base) / f"run_{run_id}_layered"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting LAYERED pipeline V4 for: {image_path.name}")
    print(f"Output directory: {run_dir}")
    
    # ------------------------------------------------------------------
    # STEP 1: CRAFT
    # ------------------------------------------------------------------
    print("\n[STEP 1] Running CRAFT (Global)...")
    detector = CraftTextDetector(cuda=False, merge_lines=True, link_threshold=0.2, text_threshold=0.6, low_text=0.35)
    craft_result = detector.detect(str(image_path))
    
    # Save crops for Gemini
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    import base64
    for region in craft_result["text_regions"]:
        b64 = region["cropped_base64"]
        if "base64," in b64: b64 = b64.split("base64,")[1]
        with open(crops_dir / f"region_{region['id']}.png", "wb") as f:
            f.write(base64.b64decode(b64))

    # ------------------------------------------------------------------
    # STEP 2: Gemini
    # ------------------------------------------------------------------
    print("\n[STEP 2] Running Gemini Analysis...")
    crop_files = sorted(list(crops_dir.glob("*.png")))
    gemini_results = []
    if crop_files:
        crop_paths_str = [str(p) for p in crop_files]
        try:
            analysis_list = analyze_text_crops_batch(crop_paths_str)
            count = min(len(crop_files), len(analysis_list))
            for i in range(count):
                crop = crop_files[i]
                gemini_results.append({
                    "region_id": crop.stem.split("_")[-1],
                    "analysis": analysis_list[i]
                })
        except Exception as e:
            print(f"Gemini failed: {e}")

    # V4.7 REFINEMENT: Logo Proximity Protection
    def refine_logo_roles(regions, g_map):
        logo_ids = []
        for r in regions:
            rid = str(r["id"])
            role = g_map.get(rid, {}).get("role", "")
            if role == "logo": logo_ids.append(r)
        
        if not logo_ids: return g_map
            
        updated_map = g_map.copy()
        for logo_r in logo_ids:
            lx, ly, lw, lh = logo_r["bbox"]["x"], logo_r["bbox"]["y"], logo_r["bbox"]["width"], logo_r["bbox"]["height"]
            for other_r in regions:
                other_rid = str(other_r["id"])
                if other_rid == str(logo_r["id"]): continue
                current_role = updated_map.get(other_rid, {}).get("role", "")
                if current_role == "logo": continue
                ox, oy, ow, oh = other_r["bbox"]["x"], other_r["bbox"]["y"], other_r["bbox"]["width"], other_r["bbox"]["height"]
                if oy > (ly + lh): gap = oy - (ly + lh)
                elif ly > (oy + oh): gap = ly - (oy + oh)
                else: gap = 0
                if gap > 20: continue
                lc, oc = lx + lw/2, ox + ow/2
                if abs(lc - oc) > max(lw, ow) * 0.5: continue
                if oh > lh * 2: continue
                print(f"  [Logo Protection] Region {other_rid} near Logo {logo_r['id']}. Encouraging 'logo' role.")
                if other_rid in updated_map: updated_map[other_rid]["role"] = "logo"
        return updated_map

    if gemini_results:
        temp_map = {str(res["region_id"]): res["analysis"] for res in gemini_results}
        fixed_map = refine_logo_roles(craft_result["text_regions"], temp_map)
        for res in gemini_results:
            rid = str(res["region_id"])
            if rid in fixed_map: res["analysis"] = fixed_map[rid]

    # ------------------------------------------------------------------
    # STEP 3: Qwen Layering
    # ------------------------------------------------------------------
    print("\n[STEP 3] Generating Layers...")
    layers_dir = run_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    
    layer_paths = []
    
    if mock_layers_dir and Path(mock_layers_dir).exists():
        print(f"Using pre-generated layers from: {mock_layers_dir}")
        import shutil
        mock_path = Path(mock_layers_dir)
        for f in sorted(mock_path.glob("*.png")):
            dest = layers_dir / f.name
            shutil.copy2(f, dest)
            layer_paths.append(str(dest))
            print(f"  > Copied layer: {dest}")
    else:
        # REAL CALL
        print("Running Qwen AI for layering...")
        try:
            saved_paths = run_qwen_layered(str(image_path), layers_dir)
            layer_paths = saved_paths
        except Exception as e:
             print(f"Qwen Failed: {e}")
             # Create dummy layer 0 so pipeline doesn't crash completely
             dummy_path = layers_dir / "0_layer_0.png"
             import shutil
             shutil.copy2(str(image_path), dummy_path)
             layer_paths = [str(dummy_path)]

    # ------------------------------------------------------------------
    # STEP 4: Layer Cleaning
    # ------------------------------------------------------------------
    print("\n[STEP 4] Cleaning Layers (Layer-Aware Pixel Erasure)...")
    orig_img = cv2.imread(str(image_path))
    cleaned_layers, cleaning_report = clean_layers(layer_paths, layers_dir, craft_result, gemini_results, orig_img.shape, original_img=orig_img)
    
    # ------------------------------------------------------------------
    # STEP 4.5: Layer Residue Detection (V4.13)
    # ------------------------------------------------------------------
    gemini_map = {str(res["region_id"]): res["analysis"] for res in gemini_results}
    
    # Find Layer 0 path from cleaned layers
    layer0_path = None
    for lp in cleaned_layers:
        if "layer_0" in Path(lp).name.lower():
            layer0_path = lp
            break
    
    residue_ids = []
    if layer0_path:
        residue_ids = detect_layer0_residue(layer0_path, craft_result["text_regions"], gemini_map, orig_img.shape)
    
    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    print("\n[Step 5] Generating JSON Report...")
    enriched_regions = []
    for region in craft_result["text_regions"]:
        rid = str(region["id"])
        r_data = region.copy()
        if "cropped_base64" in r_data: del r_data["cropped_base64"]
        r_data["gemini_analysis"] = gemini_map.get(rid, None)
        
        # V4.13: Mark residue
        if rid in residue_ids:
            r_data["layer_residue"] = True
        
        enriched_regions.append(r_data)
        
    final_report = {
        "input_image": str(image_path),
        "original_size": {"width": orig_img.shape[1], "height": orig_img.shape[0]},
        "pipeline_run_id": run_id,
        "text_detection": {"total_regions": craft_result["total_regions"], "regions": enriched_regions},
        "layer_cleaning": {"mock_source": mock_layers_dir, "layers_processed": cleaning_report}
    }
    
    report_path = run_dir / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    return str(run_dir)

if __name__ == "__main__":
    # Test
    IMAGE_PATH = r"image\IMAGE_CTA_BOX\Sparkling Family Joy.png"
    MOCK_DIR = r"outputs\IMAGE_CTA_BOX\Sparkling Family Joy_2"
    run_pipeline_layered(IMAGE_PATH, MOCK_DIR)
    pass
