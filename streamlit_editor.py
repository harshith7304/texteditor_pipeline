"""
Streamlit Canvas Editor for Pipeline V4
========================================
Advanced editing interface with canvas-like functionality:
- Visual canvas with all text elements
- Select, move, resize text regions
- Edit text content, fonts, colors
- Real-time preview
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add pipeline_v4 to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "pipeline_v4"))
sys.path.append(str(current_dir / "pipeline_v4" / "rendering"))

try:
    from streamlit_drawable_canvas import st_canvas
    from pipeline_v4.run_pipeline_text_rendering_v4 import (
        composite_layers,
        draw_background_boxes
    )
    from pipeline_v4.run_pipeline_layered_v4 import run_pipeline_layered
    from pipeline_v4.run_pipeline_box_detection_v4 import run_box_detection_pipeline
    from pipeline_v4.run_pipeline_text_rendering_v4 import render_text_layer
    from pipeline_v4.rendering.google_fonts_runtime_loader import get_font_path
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Canvas Editor - Pipeline V4",
    page_icon="✏️",
    layout="wide"
)

# Initialize session state
if 'editor_data' not in st.session_state:
    st.session_state.editor_data = {
        'report': None,
        'base_image': None,
        'run_dir': None,
        'text_regions': [],
        'selected_region': None,
        'edited_regions': {}
    }

def load_pipeline_report(run_dir):
    """Load pipeline report from run directory"""
    run_path = Path(run_dir)
    report_with_boxes = run_path / "pipeline_report_with_boxes.json"
    report_orig = run_path / "pipeline_report.json"
    report_file = report_with_boxes if report_with_boxes.exists() else report_orig
    
    if not report_file.exists():
        return None
    
    with open(report_file, "r") as f:
        return json.load(f)

def get_editable_regions(report):
    """Extract editable text regions from report"""
    regions = report.get("text_detection", {}).get("regions", [])
    editable = []
    
    for region in regions:
        gemini = region.get("gemini_analysis", {})
        role = gemini.get("role", "")
        
        # Only include editable roles
        if role in ["heading", "subheading", "body", "cta", "usp"]:
            # Skip residue regions
            if region.get("layer_residue", False):
                continue
            
            # For USP, only include if it was extracted (has box)
            if role == "usp":
                bg_box = region.get("background_box", {})
                if not bg_box.get("detected", False):
                    continue
            
            editable.append(region)
    
    return editable

def render_text_on_canvas(image, regions, edited_data=None, show_bounds=False):
    """
    Render text regions on canvas with optional edits
    edited_data: dict of {region_id: {text, font_size, color, x, y, width, height}}
    """
    if edited_data is None:
        edited_data = {}
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Get original dimensions for scaling
    orig_w = img.width
    orig_h = img.height
    
    for region in regions:
        rid = str(region["id"])
        edits = edited_data.get(rid, {})
        
        # Get original or edited values
        gemini = region.get("gemini_analysis", {})
        text = edits.get("text", gemini.get("text", ""))
        if not text.strip():
            continue
        
        # Position and size
        bbox = region["bbox"]
        x = edits.get("x", bbox["x"])
        y = edits.get("y", bbox["y"])
        w = edits.get("width", bbox["width"])
        h = edits.get("height", bbox["height"])
        
        # Font properties
        font_name = edits.get("font_name", gemini.get("primary_font", "Roboto"))
        font_weight = edits.get("font_weight", gemini.get("font_weight", 400))
        font_size = edits.get("font_size", None)
        
        # Color
        color = edits.get("color", gemini.get("text_color", "#000000"))
        
        # Calculate font size if not provided
        if font_size is None:
            # Estimate based on height
            font_size = int(h * 0.7)
        
        # Load font
        try:
            font_path = get_font_path(font_name, font_weight)
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        try:
            # Get text bbox for alignment
            text_bbox = font.getbbox(text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # Center align
            text_x = x + (w - text_w) / 2 - text_bbox[0]
            text_y = y + (h - text_h) / 2 - text_bbox[1]
            
            draw.text((text_x, text_y), text, fill=color, font=font)
            
            # Draw bounding box if requested
            if show_bounds:
                draw.rectangle([x, y, x + w, y + h], outline="#FF0000", width=2)
                # Draw region ID
                draw.text((x, y - 15), f"ID: {rid}", fill="#FF0000")
        
        except Exception as e:
            st.warning(f"Failed to render text for region {rid}: {e}")
    
    return img

def save_edited_image(base_image, regions, edited_data, run_dir):
    """Save the edited image with all modifications"""
    final_img = render_text_on_canvas(base_image, regions, edited_data, show_bounds=False)
    
    output_path = Path(run_dir) / "final_edited.png"
    final_img.save(output_path)
    return output_path

# Main UI
st.title("✏️ Canvas Editor - Text Editing & Layout")

# Sidebar for pipeline selection
with st.sidebar:
    st.header("📤 Upload & Process")
    
    # Upload new image
    st.subheader("Upload New Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to process and edit",
        key="editor_upload"
    )
    
    if uploaded_file is not None:
        # Display preview
        preview_img = Image.open(uploaded_file)
        st.image(preview_img, caption="Uploaded Image", use_container_width=True)
        
        # Save and run pipeline
        if st.button("🚀 Process Image", type="primary", use_container_width=True, key="process_new"):
            with st.spinner("Processing image..."):
                # Save uploaded file
                temp_dir = Path(tempfile.gettempdir()) / "pipeline_uploads"
                temp_dir.mkdir(exist_ok=True)
                image_path = temp_dir / uploaded_file.name
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Run pipeline
                progress_container = st.container()
                progress_container.info("🔄 Running pipeline...")
                progress_bar = progress_container.progress(0.0)
                
                try:
                    # Stage 1: Layered Pipeline
                    progress_bar.progress(0.1)
                    run_dir = run_pipeline_layered(str(image_path), mock_layers_dir=None)
                    progress_bar.progress(0.4)
                    
                    if not run_dir or not Path(run_dir).exists():
                        raise Exception("Stage 1 failed: No output directory created")
                    
                    # Stage 2: Box Detection
                    progress_bar.progress(0.5)
                    run_box_detection_pipeline(run_dir)
                    progress_bar.progress(0.7)
                    
                    # Stage 3: Text Rendering
                    progress_bar.progress(0.75)
                    
                    # Load report
                    report_with_boxes = Path(run_dir) / "pipeline_report_with_boxes.json"
                    report_orig = Path(run_dir) / "pipeline_report.json"
                    report_file = report_with_boxes if report_with_boxes.exists() else report_orig
                    
                    if not report_file.exists():
                        raise Exception("Pipeline report not found")
                    
                    with open(report_file, "r") as f:
                        report = json.load(f)
                    
                    # Get original dimensions
                    orig_w, orig_h = 1080, 1920
                    if "original_size" in report:
                        orig_w = report["original_size"]["width"]
                        orig_h = report["original_size"]["height"]
                    
                    progress_bar.progress(0.8)
                    
                    # Composite layers
                    base_img = composite_layers(str(run_dir), report)
                    if base_img is None:
                        raise Exception("Failed to composite layers")
                    
                    progress_bar.progress(0.85)
                    
                    # Draw background boxes
                    base_img = draw_background_boxes(base_img, report, orig_w, orig_h, str(run_dir))
                    
                    progress_bar.progress(0.9)
                    
                    # Render text (for final preview, but we'll use base for editing)
                    final_img = render_text_layer(base_img.copy(), report)
                    
                    # Save final image
                    out_path = Path(run_dir) / "final_composed.png"
                    final_img.save(out_path)
                    
                    progress_bar.progress(1.0)
                    
                    # Load into editor
                    st.session_state.editor_data = {
                        'report': report,
                        'base_image': base_img,
                        'run_dir': str(run_dir),
                        'text_regions': get_editable_regions(report),
                        'selected_region': None,
                        'edited_regions': {}
                    }
                    
                    progress_container.success("✅ Pipeline completed! Image loaded for editing.")
                    st.rerun()
                    
                except Exception as e:
                    progress_container.error(f"❌ Pipeline failed: {str(e)}")
                    import traceback
                    st.error(f"Error details:\n```\n{traceback.format_exc()}\n```")
    
    st.divider()
    st.header("📂 Load Existing Result")
    
    # Option 1: Select from existing runs
    pipeline_outputs = Path("pipeline_outputs")
    if pipeline_outputs.exists():
        run_dirs = sorted([d for d in pipeline_outputs.iterdir() if d.is_dir() and "layered" in d.name], 
                         reverse=True)
        
        if run_dirs:
            selected_run = st.selectbox(
                "Select Pipeline Run",
                options=[str(d) for d in run_dirs],
                format_func=lambda x: Path(x).name
            )
            
            if st.button("📥 Load Run", use_container_width=True):
                with st.spinner("Loading pipeline data..."):
                    report = load_pipeline_report(selected_run)
                    if report:
                        # Load base image
                        run_path = Path(selected_run)
                        final_path = run_path / "final_composed.png"
                        
                        if final_path.exists():
                            # Reconstruct base (without text) for editing
                            base_img = composite_layers(str(run_path), report)
                            if base_img:
                                base_img = draw_background_boxes(base_img, report, 
                                                               report.get("original_size", {}).get("width", 1080),
                                                               report.get("original_size", {}).get("height", 1920),
                                                               str(run_path))
                                
                                st.session_state.editor_data = {
                                    'report': report,
                                    'base_image': base_img,
                                    'run_dir': selected_run,
                                    'text_regions': get_editable_regions(report),
                                    'selected_region': None,
                                    'edited_regions': {}
                                }
                                st.success("✅ Pipeline data loaded!")
                                st.rerun()
                        else:
                            st.error("Final composed image not found. Run pipeline first.")
                    else:
                        st.error("Failed to load pipeline report.")
        else:
            st.info("No pipeline runs found. Run the pipeline first.")
    else:
        st.warning("pipeline_outputs directory not found.")

# Main editor area
# Initialize edited_regions to avoid NameError
edited_regions = st.session_state.editor_data.get('edited_regions', {})

if st.session_state.editor_data['report'] is None:
    st.info("👈 Load a pipeline run from the sidebar to start editing")
else:
    editor_data = st.session_state.editor_data
    base_image = editor_data['base_image']
    regions = editor_data['text_regions']
    # Get edited_regions from session state - this is our working copy
    edited_regions = editor_data.get('edited_regions', {}).copy()
    
    # Two-column layout: Canvas and Controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 Interactive Canvas")
        
        # Canvas size
        canvas_width = base_image.width
        canvas_height = base_image.height
        
        # Scale down if too large for display
        max_display_width = 800
        if canvas_width > max_display_width:
            scale_factor = max_display_width / canvas_width
            display_width = int(canvas_width * scale_factor)
            display_height = int(canvas_height * scale_factor)
        else:
            display_width = canvas_width
            display_height = canvas_height
            scale_factor = 1.0
        
        # Render preview with bounds
        preview_img = render_text_on_canvas(base_image, regions, edited_regions, show_bounds=True)
        
        # Resize for canvas display
        if scale_factor != 1.0:
            preview_img_display = preview_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        else:
            preview_img_display = preview_img
        
        # Convert to base64 for canvas
        import base64
        from io import BytesIO
        buffered = BytesIO()
        preview_img_display.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode()
        
        # Drawable canvas for interaction
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=Image.open(BytesIO(base64.b64decode(img_data))) if img_data else None,
            update_streamlit=True,
            width=display_width,
            height=display_height,
            drawing_mode="transform",  # Allow moving/transforming
            point_display_radius=5 if st.session_state.get("show_points", False) else 0,
            key="canvas"
        )
        
        # Show full-size preview below
        st.image(preview_img, caption="Full Preview (with bounds)", use_container_width=True)
        
        # Region selector
        if regions:
            st.subheader("📍 Select Text Region")
            region_options = {
                f"Region {r['id']} - {r.get('gemini_analysis', {}).get('text', '')[:30]}...": r['id']
                for r in regions
            }
            
            selected_label = st.selectbox(
                "Choose a region to edit",
                options=list(region_options.keys()),
                index=0 if editor_data['selected_region'] is None else None
            )
            
            selected_id = region_options[selected_label]
            editor_data['selected_region'] = selected_id
            
            # Find selected region
            selected_region = next((r for r in regions if r['id'] == selected_id), None)
            
            if selected_region:
                # Highlight selected region
                highlight_img = base_image.copy()
                draw = ImageDraw.Draw(highlight_img)
                bbox = selected_region["bbox"]
                draw.rectangle(
                    [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]],
                    outline="#00FF00", width=3
                )
                # Render all text
                highlight_img = render_text_on_canvas(highlight_img, regions, edited_regions, show_bounds=False)
                st.image(highlight_img, caption="Selected Region Highlighted", use_container_width=True)
    
    with col2:
        st.subheader("⚙️ Edit Properties")
        
        if editor_data['selected_region'] and selected_region:
            rid = str(selected_region['id'])
            gemini = selected_region.get("gemini_analysis", {})
            current_edits = edited_regions.get(rid, {})
            
            # Text content
            st.write("**Text Content**")
            current_text = current_edits.get("text", gemini.get("text", ""))
            new_text = st.text_area("Text", value=current_text, key=f"text_{rid}", height=100)
            if new_text != current_text:
                if rid not in edited_regions:
                    edited_regions[rid] = {}
                edited_regions[rid]["text"] = new_text
                # Update session state immediately
                st.session_state.editor_data['edited_regions'] = edited_regions
            
            st.divider()
            
            # Position (with drag helper)
            st.write("**Position**")
            bbox = selected_region["bbox"]
            
            # Show current position
            st.caption(f"Current: ({bbox['x']}, {bbox['y']})")
            
            col_x, col_y = st.columns(2)
            with col_x:
                new_x = st.number_input("X", value=current_edits.get("x", bbox["x"]), 
                                       min_value=0, max_value=base_image.width, 
                                       step=5, key=f"x_{rid}")
            with col_y:
                new_y = st.number_input("Y", value=current_edits.get("y", bbox["y"]), 
                                       min_value=0, max_value=base_image.height, 
                                       step=5, key=f"y_{rid}")
            
            # Quick move buttons
            col_l, col_r, col_u, col_d = st.columns(4)
            with col_l:
                if st.button("←", key=f"left_{rid}"):
                    new_x = max(0, new_x - 10)
            with col_r:
                if st.button("→", key=f"right_{rid}"):
                    new_x = min(base_image.width - bbox["width"], new_x + 10)
            with col_u:
                if st.button("↑", key=f"up_{rid}"):
                    new_y = max(0, new_y - 10)
            with col_d:
                if st.button("↓", key=f"down_{rid}"):
                    new_y = min(base_image.height - bbox["height"], new_y + 10)
            
            if new_x != current_edits.get("x", bbox["x"]) or new_y != current_edits.get("y", bbox["y"]):
                if rid not in edited_regions:
                    edited_regions[rid] = {}
                edited_regions[rid]["x"] = int(new_x)
                edited_regions[rid]["y"] = int(new_y)
                # Update session state
                st.session_state.editor_data['edited_regions'] = edited_regions
                st.rerun()
            
            # Size (with resize helper)
            st.write("**Size**")
            col_w, col_h = st.columns(2)
            with col_w:
                new_w = st.number_input("Width", value=current_edits.get("width", bbox["width"]), 
                                       min_value=10, max_value=base_image.width,
                                       step=5, key=f"w_{rid}")
            with col_h:
                new_h = st.number_input("Height", value=current_edits.get("height", bbox["height"]), 
                                       min_value=10, max_value=base_image.height,
                                       step=5, key=f"h_{rid}")
            
            # Quick resize buttons
            col_smaller, col_larger = st.columns(2)
            with col_smaller:
                if st.button("🔽 Smaller", key=f"smaller_{rid}"):
                    new_w = max(10, int(new_w * 0.9))
                    new_h = max(10, int(new_h * 0.9))
            with col_larger:
                if st.button("🔼 Larger", key=f"larger_{rid}"):
                    new_w = min(base_image.width, int(new_w * 1.1))
                    new_h = min(base_image.height, int(new_h * 1.1))
            
            if new_w != current_edits.get("width", bbox["width"]) or new_h != current_edits.get("height", bbox["height"]):
                if rid not in edited_regions:
                    edited_regions[rid] = {}
                edited_regions[rid]["width"] = int(new_w)
                edited_regions[rid]["height"] = int(new_h)
                # Update session state
                st.session_state.editor_data['edited_regions'] = edited_regions
                st.rerun()
            
            st.divider()
            
            # Font properties
            st.write("**Font**")
            current_font = current_edits.get("font_name", gemini.get("primary_font", "Roboto"))
            font_options = ["Roboto", "Poppins", "Inter", "Manrope", "Plus Jakarta Sans", "Oswald", "Bebas Neue", "Anton"]
            new_font = st.selectbox("Font Family", options=font_options, 
                                   index=font_options.index(current_font) if current_font in font_options else 0,
                                   key=f"font_{rid}")
            
            if new_font != current_font:
                if rid not in edited_regions:
                    edited_regions[rid] = {}
                edited_regions[rid]["font_name"] = new_font
                # Update session state
                st.session_state.editor_data['edited_regions'] = edited_regions
            
            current_weight = current_edits.get("font_weight", gemini.get("font_weight", 400))
            weight_options = [300, 400, 500, 600, 700, 800]
            new_weight = st.selectbox("Font Weight", options=weight_options,
                                    index=weight_options.index(current_weight) if current_weight in weight_options else 1,
                                    key=f"weight_{rid}")
            
            if new_weight != current_weight:
                if rid not in edited_regions:
                    edited_regions[rid] = {}
                edited_regions[rid]["font_weight"] = new_weight
                # Update session state
                st.session_state.editor_data['edited_regions'] = edited_regions
            
            new_size = st.number_input("Font Size", value=current_edits.get("font_size", int(bbox["height"] * 0.7)),
                                      min_value=8, max_value=200, key=f"size_{rid}")
            
            if new_size != current_edits.get("font_size", int(bbox["height"] * 0.7)):
                if rid not in edited_regions:
                    edited_regions[rid] = {}
                edited_regions[rid]["font_size"] = int(new_size)
                # Update session state
                st.session_state.editor_data['edited_regions'] = edited_regions
            
            st.divider()
            
            # Color
            st.write("**Color**")
            current_color = current_edits.get("color", gemini.get("text_color", "#000000"))
            new_color = st.color_picker("Text Color", value=current_color, key=f"color_{rid}")
            
            if new_color != current_color:
                if rid not in edited_regions:
                    edited_regions[rid] = {}
                edited_regions[rid]["color"] = new_color
                # Update session state
                st.session_state.editor_data['edited_regions'] = edited_regions
            
            st.divider()
            
            # Action buttons
            if st.button("🔄 Reset Region", use_container_width=True, key=f"reset_{rid}"):
                if rid in edited_regions:
                    del edited_regions[rid]
                # Update session state
                st.session_state.editor_data['edited_regions'] = edited_regions
                st.rerun()
        
        else:
            st.info("Select a region to edit")
        
        st.divider()
        
        # Global actions
        st.subheader("💾 Actions")
        
        if st.button("🔄 Refresh Preview", use_container_width=True):
            st.rerun()
        
        if st.button("💾 Save Edited Image", use_container_width=True, type="primary"):
            with st.spinner("Saving..."):
                output_path = save_edited_image(
                    base_image, 
                    regions, 
                    edited_regions,
                    editor_data['run_dir']
                )
                st.success(f"✅ Saved to: {output_path}")
                
                # Download button
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Edited Image",
                        data=f.read(),
                        file_name="final_edited.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        if st.button("🗑️ Clear All Edits", use_container_width=True):
            edited_regions.clear()
            # Update session state
            st.session_state.editor_data['edited_regions'] = edited_regions
            st.rerun()
        
        # Show edit summary
        if edited_regions:
            st.divider()
            st.write(f"**📝 {len(edited_regions)} region(s) edited**")
            for rid, edits in edited_regions.items():
                st.write(f"- Region {rid}: {len(edits)} property(ies) changed")

# Ensure edited_regions is always available in session state
# (Already handled above with immediate updates, but this is a safety check)
if st.session_state.editor_data['report'] is not None and 'edited_regions' not in st.session_state.editor_data:
    st.session_state.editor_data['edited_regions'] = {}

