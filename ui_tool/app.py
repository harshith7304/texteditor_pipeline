import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from pathlib import Path
import json
import backend # Local import

def main():
    st.set_page_config(layout="wide", page_title="Pipeline Interaction Test")
    
    # --- PIPELINE IMPORTS ---
    import sys
    import os
    # Add root to path to find pipeline_v4
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
        
    try:
        from pipeline_v4.run_pipeline_layered_v4 import run_pipeline_layered
        from pipeline_v4.run_pipeline_box_detection_v4 import run_box_detection_pipeline
        from pipeline_v4.rendering.google_fonts_runtime_loader import get_font_path
    except ImportError as e:
        st.error(f"Pipeline Import Error: {e}")
        return

    st.sidebar.title("Pipeline Interaction Lab 🧪")
    
    # 0. NEW IMAGE UPLOAD
    uploaded_file = st.sidebar.file_uploader("Upload New Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        # Check if already processed to avoid re-running on every interactions
        # We use a simple hash or name check
        if st.sidebar.button("🚀 Run Pipeline on Upload"):
            with st.spinner("Saving image..."):
                # Save to temp
                uploads_dir = root_dir / "image" / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                file_path = uploads_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
            status_container = st.empty()
            
            try:
                # 1. Layering
                status_container.info("⏳ Running Layered Pipeline (CRAFT + Gemini + Qwen)... this takes ~30s")
                run_dir = run_pipeline_layered(str(file_path))
                
                # 2. Box Detection
                status_container.info("📦 Detecting Background Boxes...")
                run_box_detection_pipeline(run_dir)
                
                # 3. Text Rendering (Auto-render so image appears immediately)
                status_container.info("✏️ Rendering Text...")
                backend.render_with_pipeline(run_dir)
                
                # 4. Complete
                status_container.success("✅ Pipeline Complete!")
                import time
                time.sleep(1)
                status_container.empty()
                
                # 5. Auto-Select
                run_id = Path(run_dir).name.split("_")[1] # run_123_layered -> 123
                # We need to refresh the list, backend.list_pipeline_runs() is called below
                # Force reload by rerun
                # st.session_state.current_run_id = f"run_{run_id}_layered" # REMOVED: Proactive set causes skip of reset logic
                # Actually main() re-reads runs. We just need to ensure selectbox defaults to new one.
                # Store preference in session state
                st.session_state.auto_select_run = f"run_{run_id}_layered"
                st.rerun()
                
            except Exception as e:
                st.error(f"Pipeline Failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.sidebar.markdown("---")
    
    # 1. Select Run
    runs = backend.list_pipeline_runs()
    if not runs:
        st.error("No pipeline runs found in `pipeline_outputs/`.")
        return
        
    run_options = {r["id"]: r for r in runs}
    
    # Auto-select logic
    default_idx = 0
    if "auto_select_run" in st.session_state and st.session_state.auto_select_run in run_options:
        keys = list(run_options.keys())
        default_idx = keys.index(st.session_state.auto_select_run)
        
    selected_run_id = st.sidebar.selectbox("Select Pipeline Run", list(run_options.keys()), index=default_idx)
    
    run_details = run_options[selected_run_id]
    st.sidebar.info(f"Path: {run_details['path']}")
    
    # 2. Load Data
    data = backend.load_run_data(selected_run_id)
    if not data or "error" in data:
        st.error(f"Failed to load run data: {data.get('error')}")
        return
        
    report = data["report"]
    bg_image = data["background_image"]
    
    # --- INJECT GOOGLE FONTS CSS FOR BROWSER RENDERING ---
    # Fabric.js canvas runs in the browser and needs fonts loaded via CSS, not server-side
    text_regions = report.get("text_detection", {}).get("regions", [])
    unique_fonts = {}  # font_name -> set of weights
    for region in text_regions:
        gemini = region.get("gemini_analysis", {})
        if gemini:
            font_name = gemini.get("primary_font", "Roboto")
            font_weight = gemini.get("font_weight", 400)
            if font_name not in unique_fonts:
                unique_fonts[font_name] = set()
            unique_fonts[font_name].add(font_weight)
    
    # Build Google Fonts URL
    if unique_fonts:
        font_specs = []
        for font_name, weights in unique_fonts.items():
            weight_str = ";".join([f"wght@{w}" for w in sorted(weights)])
            family_encoded = font_name.replace(" ", "+")
            font_specs.append(f"family={family_encoded}:wght@{','.join(map(str, sorted(weights)))}")
        
        google_fonts_url = f"https://fonts.googleapis.com/css2?{'&'.join(font_specs)}&display=swap"
        
        # Inject the font CSS
        st.markdown(f'''
        <link href="{google_fonts_url}" rel="stylesheet">
        <style>
            /* Force canvas text to use the correct fonts */
            .canvas-container, canvas, .upper-canvas {{
                font-family: inherit !important;
            }}
        </style>
        ''', unsafe_allow_html=True)
    # --- END FONT INJECTION ---
    
    # --- DEBUG PANEL ---
    with st.sidebar.expander("🔍 Debug: Full JSON Report", expanded=False):
        import json
        st.json(report)
        
        st.markdown("---")
        st.markdown("### Font Weight Summary")
        text_regions = report.get("text_detection", {}).get("regions", [])
        for region in text_regions:
            rid = region.get("id", "?")
            gemini = region.get("gemini_analysis", {})
            if not gemini: continue
            
            role = gemini.get("role", "?")
            text = gemini.get("text", "")[:30]
            font = gemini.get("primary_font", "?")
            weight = gemini.get("font_weight", "?")
            residue = region.get("layer_residue", False)
            
            status = "🔴 RESIDUE" if residue else "🟢 OK"
            st.markdown(f"**R{rid}** ({role}): `{font}` @ **{weight}** {status}")
            st.caption(f"'{text}...'")
    # --- END DEBUG ---
    
    # 3. Canvas Config
    # Sidebar width override
    # 3. Canvas Config
    st.sidebar.markdown("---")
    
    # Split controls: Image Scale vs Canvas Size
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        img_display_width = st.number_input("Image Width", 300, 1200, 480, step=20)
    with col_s2:
        canvas_width = st.number_input("Canvas Width", 500, 2500, 1400, step=50) # Large default
    
    # Calculate scale based on ORIGINAL image dimensions vs Target Display Width
    orig_w = data.get("original_size", {}).get("width", 1080)
    orig_h = data.get("original_size", {}).get("height", 1920)
    
    # Aspect Ratio
    aspect_ratio = orig_h / orig_w
    
    # Image Display Dimensions
    img_display_height = int(img_display_width * aspect_ratio)
    
    # Canvas Dimensions (Workspace)
    canvas_height = st.sidebar.number_input("Canvas Height", 500, 2500, 1000, step=50)
    
    scale_x = img_display_width / orig_w
    scale_y = img_display_height / orig_h
    
    st.sidebar.markdown(f"**Original**: {orig_w}x{orig_h}")
    st.sidebar.markdown(f"**Display**: {img_display_width}x{img_display_height}")
    st.sidebar.markdown(f"**Scale**: {scale_x:.4f}")

    # --- HELPERS ---
    import base64
    from io import BytesIO
    from PIL import ImageFont, ImageDraw
    import time # Imported for sleep above
    import math

    def image_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    # Default fallback font path
    DEFAULT_FONT = "Roboto-400.ttf"
    FALLBACK_FONT_PATH = root_dir / "fonts" / DEFAULT_FONT

    def get_dynamic_font_path(font_name, font_weight):
        """
        Get the correct font file for the given font name and weight.
        Falls back to bundled Roboto if Google Fonts fails.
        """
        try:
            return get_font_path(font_name, font_weight)
        except Exception as e:
            print(f"Font loading failed for {font_name}@{font_weight}: {e}")
            return str(FALLBACK_FONT_PATH)

    def calculate_font_size(text, box_h, box_w, font_path=None):
        """
        Calculate max font size where text fits in box (handling wrapping).
        """
        if not text: return 10
        
        low, high = 1, int(box_h)
        best = 10
        
        # Optimization: Longest word width check (must fit horizontally)
        words = text.split()
        max_word = max(words, key=len) if words else text
        
        while low <= high:
            mid = (low + high) // 2
            try:
                font = ImageFont.truetype(str(font_path or FALLBACK_FONT_PATH), mid)
                
                # 1. Check if longest word fits width
                # getlength is more accurate than getbbox for width
                word_w = font.getlength(max_word)
                if word_w > box_w:
                    high = mid - 1
                    continue
                
                # 2. Check total wrapping height
                # Approximate lines = Total Text Width / Box Width
                total_w = font.getlength(text)
                num_lines = max(1, math.ceil(total_w / box_w))
                
                # Heuristic: Add 10% overflow buffer just in case
                # Line height typically ~1.2 * font_size
                required_h = num_lines * (mid * 1.3) 
                
                if required_h <= box_h:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            except:
                break
        
        return best

    # --- PREPARE BACKGROUND OBJECT ---
    # Resized BG for display (Needed for Base64 injection) relative to Image Display Size
    bg_display = bg_image.resize((img_display_width, img_display_height))
    
    buffered = BytesIO()
    bg_display.save(buffered, format="PNG")
    bg_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    bg_object = {
        "type": "image",
        "version": "4.4.0",
        "originX": "left",
        "originY": "top",
        "left": 0,
        "top": 0,
        "width": img_display_width,
        "height": img_display_height,
        "fill": "rgb(0,0,0)",
        "stroke": None,
        "strokeWidth": 0,
        "strokeDashArray": None,
        "strokeLineCap": "butt",
        "strokeDashOffset": 0,
        "strokeLineJoin": "miter",
        "strokeMiterLimit": 4,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "flipX": False,
        "flipY": False,
        "opacity": 1,
        "shadow": None,
        "visible": True,
        "clipTo": None,
        "backgroundColor": "",
        "fillRule": "nonzero",
        "paintFirst": "fill",
        "globalCompositeOperation": "source-over",
        "transformMatrix": None,
        "skewX": 0,
        "skewY": 0,
        "src": f"data:image/png;base64,{bg_b64}",
        "crossOrigin": None,
        "selectable": False, # Lock background
        "evented": False     # Ignore clicks
    }

    # ... (After Sidebar Config) ...
    
    # State Management for Re-renders
    if "canvas_version" not in st.session_state:
        st.session_state.canvas_version = 0
    
    # --- PREPARE BASE OBJECTS (From Report) ---
    base_objects = []
    
    # 1. Background
    bg_object["u_id"] = "background"
    base_objects.append(bg_object)
    
    text_regions = report.get("text_detection", {}).get("regions", [])
    
    # 2. Extracted Boxes
    for i, region in enumerate(text_regions):
        rid = region.get("id", i)
        bg_box = region.get("background_box", {})
        
        # ... [Existing Box Logic] ...
        if bg_box.get("detected"):
            bbox = bg_box["bbox"]
            # ... (Box Path Resolution) ...
            extracted_path = bg_box.get("extracted_image")
            
            final_obj = None
            if extracted_path:
                full_path = Path(data["run_dir"]) / extracted_path
                if full_path.exists():
                    # ... (Load Image) ...
                    box_img = Image.open(full_path).convert("RGBA")
                    box_b64 = image_to_base64(box_img)
                    
                    if "layer_bbox" in bg_box:
                        lb = bg_box["layer_bbox"]
                        bg_w, bg_h = bg_image.size
                        sx_layer = img_display_width / bg_w
                        sy_layer = img_display_height / bg_h
                        f_l, f_t = lb["x"]*sx_layer, lb["y"]*sy_layer
                        f_w, f_h = lb["width"]*sx_layer, lb["height"]*sy_layer
                    else:
                        f_l, f_t = bbox["x"]*scale_x, bbox["y"]*scale_y
                        f_w, f_h = bbox["width"]*scale_x, bbox["height"]*scale_y

                    final_obj = {
                        "type": "image",
                        "version": "4.4.0",
                        "originX": "left", "originY": "top",
                        "left": f_l, "top": f_t, "width": f_w, "height": f_h,
                        "src": f"data:image/png;base64,{box_b64}",
                        "scaleX": 1, "scaleY": 1,
                        "opacity": 1, "selectable": True 
                    }
            
            if not final_obj:
                # Fallback Rect
                color = bg_box.get("color", "#000000")
                final_obj = {
                    "type": "rect",
                    "left": bbox["x"]*scale_x, "top": bbox["y"]*scale_y,
                    "width": bbox["width"]*scale_x, "height": bbox["height"]*scale_y,
                    "fill": color
                }
            
            final_obj["u_id"] = f"box_{rid}"
            base_objects.append(final_obj)

    # 3. Text Objects & Editor Form
    text_updates = {}
    debug_shown = False
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("")
    with st.sidebar.form("text_editor_form"):
        for i, region in enumerate(text_regions):
            gemini = region.get("gemini_analysis", {})
            role = gemini.get("role", "body")
            
            # V4.13 FIX: Skip if text is residue on background
            if region.get("layer_residue", False):
                continue
            
            # V4.12 FIX: Exclude Preserved/Protected roles from Editor
            # Matches backend logic: IF preserved in BG, DO NOT allow editing/rendering on top
            if role in ["product_text", "logo", "icon", "label", "ui_element"]:
                continue
            
            # Hybrid USP: Only show if extracted
            bg_box = region.get("background_box", {})
            if role == "usp" and not bg_box.get("detected", False):
                 continue
                
            rid = region.get("id", i)
            orig_text = gemini.get("text", "")
            
            # Text Input
            new_text = st.text_area(f"Text {rid} ({role})", value=orig_text, height=70)
            text_updates[f"text_{rid}"] = new_text
            
            # Base Object Creation
            bbox = region["bbox"]
            color = gemini.get("text_color", "#000000")
            canvas_box_h = bbox["height"] * scale_y
            canvas_box_w = bbox["width"] * scale_x
            
            # --- DEBUG INFO FOR FIRST TEXT (RE-ADDED) ---
            if not debug_shown:
                debug_shown = True
                with st.sidebar.expander("🕵️‍♀️ Font Debug (First Region)", expanded=True):
                    st.info(f"App Version: Wrapp-Aware Fix")
                    
                    st.write(f"**Font Path**: `{FALLBACK_FONT_PATH}`")
                    st.write(f"**Exists**: `{FALLBACK_FONT_PATH.exists()}`")
                    
                    # Scaling Debug
                    st.write("---")
                    st.write(f"**Orig Img W**: {orig_w}")
                    st.write(f"**Display W**: {img_display_width}")
                    st.write(f"**Scale X**: {scale_x:.4f}")
                    
                    # Test Calculation
                    try:
                        test_font = ImageFont.truetype(str(FALLBACK_FONT_PATH), 10)
                        st.success("Font loaded successfully!")
                    except Exception as e:
                        st.error(f"Font load failed: {e}")
                    
                    st.write("---")
                    st.write(f"**Target Box**: {canvas_box_w:.1f}w x {canvas_box_h:.1f}h")
                    st.write(f"**Text**: '{orig_text[:20]}...'")
                    
                    # Manual Calc Trace - Use Gemini's font
                    debug_font_name = gemini.get("primary_font", "Roboto")
                    debug_font_weight = gemini.get("font_weight", 400)
                    debug_font_path = get_dynamic_font_path(debug_font_name, debug_font_weight)
                    
                    st.write(f"**Dynamic Font**: `{debug_font_name}@{debug_font_weight}`")
                    st.write(f"**Font File**: `{debug_font_path}`")
                    
                    calc_size = calculate_font_size(orig_text, canvas_box_h, canvas_box_w, debug_font_path)
                    st.write(f"-> **Calculated Size**: {calc_size}")
                    
                    # Check metrics at this size
                    try:
                        f = ImageFont.truetype(str(debug_font_path), calc_size)
                        # Use getlength for width check as per new logic
                        w = f.getlength(orig_text) # Single line estimation for debug
                        
                        # For height, we need the messy bbox
                        l, t, r, b = f.getbbox(orig_text)
                        h = b - t
                        
                        st.write(f"**Actual Text W**: {w:.1f}")
                        st.write(f"**Actual Text H**: {h:.1f}")
                        
                        if w > 0:
                            st.write(f"**Width Util**: {w/canvas_box_w*100:.1f}%")
                        if h > 0:
                            st.write(f"**Height Util**: {h/canvas_box_h*100:.1f}%")
                            
                        # Word Check
                        words = orig_text.split()
                        if words:
                            max_word = max(words, key=len)
                            mw_w = f.getlength(max_word)
                            st.write(f"**Longest Word '{max_word}' W**: {mw_w:.1f}")
                            if mw_w > canvas_box_w:
                                st.error(f"⚠️ Word overflow! {mw_w:.1f} > {canvas_box_w:.1f}")
                    except Exception as e:
                        st.write(f"Metric Check Error: {e}")
            # ---------------------------------
            
            # Get font info from Gemini analysis
            font_name = gemini.get("primary_font", "Roboto")
            font_weight = gemini.get("font_weight", 400)
            
            # Load the correct font dynamically
            dynamic_font_path = get_dynamic_font_path(font_name, font_weight)
            
            font_size = calculate_font_size(orig_text, canvas_box_h, canvas_box_w, dynamic_font_path)
            
            base_objects.append({
                "type": "textbox",
                "u_id": f"text_{rid}", # Unique ID for matching
                "text": orig_text, # Initial text
                "left": bbox["x"] * scale_x,
                "top": bbox["y"] * scale_y,
                "width": bbox["width"] * scale_x,
                "fontSize": font_size,
                "fontFamily": font_name,  # Use correct font family from Gemini
                "fontWeight": font_weight,  # Pass weight for CSS rendering
                "fill": color,
                "backgroundColor": "transparent"
            })
            
        update_clicked = st.form_submit_button("Update Text")

    # --- STATE MANAGEMENT ---
    
    # Check if run changed, reset state
    if "current_run_id" not in st.session_state or st.session_state.current_run_id != selected_run_id:
        st.session_state.current_run_id = selected_run_id
        st.session_state.canvas_version = 0
        st.session_state.active_objects = base_objects # Initialize with defaults
        st.session_state.last_canvas_state = None
        
    # Manual Reset Button
    if st.sidebar.button("Reset Layout"):
        st.session_state.active_objects = base_objects
        st.session_state.canvas_version += 1
        st.session_state.last_canvas_state = None
        st.success("Layout reset to defaults.")
        # We need to rerun to reflect this immediately
        st.rerun()

    final_drawing_objects = st.session_state.active_objects
    
    # --- MERGE POSITIONS ON CLICK ---
    # Only run if we have a valid previous state
    if update_clicked and st.session_state.last_canvas_state is not None:
        current_objects = st.session_state.last_canvas_state["objects"]
        merged = []
        
        # Base schema is st.session_state.active_objects
        # We assume strict index matching for now as established
        base_list = st.session_state.active_objects
        
        if len(current_objects) == len(base_list):
             for idx, base_obj in enumerate(base_list):
                 curr_obj = current_objects[idx]
                 
                 # Verify Type
                 if base_obj.get("type") != curr_obj.get("type"):
                     # Fallback to base (don't merge mismatch)
                     merged.append(base_obj)
                     continue
                 
                 # Copy Positions from Canvas back to State
                 preserved_props = ["left", "top", "width", "height", "scaleX", "scaleY", "angle"]
                 new_obj = base_obj.copy()
                 for p in preserved_props:
                     if p in curr_obj:
                         new_obj[p] = curr_obj[p]
                 
                 merged.append(new_obj)
             
             st.session_state.active_objects = merged
             st.session_state.canvas_version += 1
             st.success(f"Positions synced! (v{st.session_state.canvas_version})")
        else:
             st.warning(f"Object mismatch ({len(current_objects)} vs {len(base_list)}). Update skipped to prevent data loss.")

    # --- APPLY TEXT UPDATES (ALWAYS) ---
    updated_objects = []
    for obj in st.session_state.active_objects:
        obj_copy = obj.copy()
        uid = obj_copy.get("u_id")
        
        if uid and uid in text_updates and obj_copy["type"] == "textbox":
             # Override text with whatever is in the sidebar text_area
             obj_copy["text"] = text_updates[uid]
        
        updated_objects.append(obj_copy)
    
    final_drawing_objects = updated_objects

    # --- PIPELINE PREVIEW (Shows actual rendered output) ---
    st.subheader("🎨 Pipeline Preview")
    
    run_dir = data.get("run_dir")
    
    # Check for existing final_composed.png
    final_composed_path = Path(run_dir) / "final_composed.png" if run_dir else None
    
    # Create text updates dict from sidebar form
    text_edit_updates = {}
    for key, value in text_updates.items():
        # Extract region ID from "text_1", "text_2" format
        rid = key.replace("text_", "")
        text_edit_updates[rid] = value
    
    col_preview, col_controls = st.columns([4, 1])
    
    with col_controls:
        st.markdown("### Actions")
        
        # Re-render button
        if st.button("🔄 Re-render with Edits", type="primary"):
            if run_dir:
                with st.spinner("Rendering with pipeline..."):
                    rendered_img = backend.render_with_pipeline(run_dir, text_edit_updates)
                    if rendered_img:
                        st.success("✅ Rendered!")
                        st.session_state.pipeline_preview_updated = True
                        st.rerun()
                    else:
                        st.error("Rendering failed")
            else:
                st.error("No run directory found")
        
        # Download button
        if final_composed_path and final_composed_path.exists():
            with open(final_composed_path, "rb") as f:
                st.download_button(
                    label="📥 Download Final",
                    data=f,
                    file_name="final_composed.png",
                    mime="image/png"
                )
    
    with col_preview:
        if final_composed_path and final_composed_path.exists():
            # Show the actual pipeline-rendered image
            st.image(str(final_composed_path), caption="Pipeline Output (final_composed.png)", use_container_width=True)
        else:
            st.warning("No final_composed.png found. Click 'Re-render with Edits' to generate.")
    
    # --- CANVAS EDITOR (for positioning) ---
    st.markdown("---")
    with st.expander("🔧 Advanced: Canvas Editor (for positioning)", expanded=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.subheader("Interactive Editor")
            # Dynamic Key to force update when needed
            c_key = f"canvas_{selected_run_id}_v{st.session_state.canvas_version}"
            
            # Note: background_image in st_canvas is broken in newer Streamlit versions
            # The rendered result is shown in Pipeline Preview above
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#000000",
                background_color="#eeeeee",
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="freedraw",
                initial_drawing={"version": "4.4.0", "objects": []},
                key=c_key,
            )
            
            # Save state for next run
            if canvas_result.json_data:
                st.session_state.last_canvas_state = canvas_result.json_data
            
        with col2:
            st.subheader("Data Inspector")
            if canvas_result.json_data:
                objects = canvas_result.json_data["objects"]
                # Filter out the background image from inspector
                editable_objects = [o for o in objects if o.get("type") != "image"]
                
                st.write(f"Active Objects: {len(editable_objects)}")
                # Show modified objects
                for i, obj in enumerate(editable_objects):
                    st.caption(f"Object {i} ({obj['type']})")
                    st.json({
                        "u_id": obj.get("u_id", "MISSING"),
                        "text": obj.get("text", "N/A"),
                        "left": int(obj["left"]),
                        "top": int(obj["top"]),
                        "width": int(obj["width"] * obj.get("scaleX", 1)),
                        "height": int(obj["height"] * obj.get("scaleY", 1)),
                        "fill": obj.get("fill", "N/A")
                    })

if __name__ == "__main__":
    main()
