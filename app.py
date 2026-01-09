"""
Unified Pipeline App - Upload, Process & Edit
============================================
Single app for complete workflow:
1. Upload image
2. Process (CRAFT + Gemini + Qwen + Box Detection)
3. Edit text and CTA boxes
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
    from pipeline_v4.run_pipeline_text_rendering_v4 import (
        composite_layers,
        draw_background_boxes,
        render_text_layer
    )
    from pipeline_v4.run_pipeline_layered_v4 import run_pipeline_layered
    from pipeline_v4.run_pipeline_box_detection_v4 import run_box_detection_pipeline
    from pipeline_v4.rendering.google_fonts_runtime_loader import get_font_path
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Image Editor - Pipeline V4",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'pipeline_data' not in st.session_state:
    st.session_state.pipeline_data = {
        'report': None,
        'base_image': None,
        'run_dir': None,
        'text_regions': [],
        'box_regions': [],
        'selected_text': None,
        'selected_box': None,
        'edited_text': {},
        'edited_boxes': {}
    }

def get_editable_regions(report):
    """Extract editable text regions from report"""
    regions = report.get("text_detection", {}).get("regions", [])
    editable = []
    
    for region in regions:
        gemini = region.get("gemini_analysis", {})
        role = gemini.get("role", "")
        
        if role in ["heading", "subheading", "body", "cta", "usp"]:
            if region.get("layer_residue", False):
                continue
            if role == "usp":
                bg_box = region.get("background_box", {})
                if not bg_box.get("detected", False):
                    continue
            editable.append(region)
    
    return editable

def get_editable_boxes(report):
    """Extract editable CTA boxes from report"""
    regions = report.get("text_detection", {}).get("regions", [])
    boxes = []
    
    for region in regions:
        bg_box = region.get("background_box", {})
        if bg_box.get("detected", False):
            boxes.append({
                'region_id': region['id'],
                'bbox': bg_box['bbox'],
                'color': bg_box.get('color', '#000000'),
                'text': region.get('gemini_analysis', {}).get('text', '')
            })
    
    return boxes

def render_editable_canvas(image, text_regions, box_regions, edited_text, edited_boxes, show_bounds=False):
    """Render canvas with text and boxes, applying edits"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Draw boxes first (background)
    for box in box_regions:
        bid = str(box['region_id'])
        edits = edited_boxes.get(bid, {})
        
        bbox = edits.get('bbox', box['bbox'])
        color = edits.get('color', box['color'])
        
        # Parse color
        try:
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            else:
                r, g, b = 0, 0, 0
        except:
            r, g, b = 0, 0, 0
        
        # Draw box
        x, y = bbox['x'], bbox['y']
        w, h = bbox['width'], bbox['height']
        draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 200), outline=(r, g, b), width=2)
        
        if show_bounds:
            draw.text((x, y - 15), f"Box {bid}", fill="#FF00FF")
    
    # Draw text on top
    for region in text_regions:
        rid = str(region["id"])
        edits = edited_text.get(rid, {})
        
        gemini = region.get("gemini_analysis", {})
        text = edits.get("text", gemini.get("text", ""))
        if not text.strip():
            continue
        
        bbox = region["bbox"]
        x = edits.get("x", bbox["x"])
        y = edits.get("y", bbox["y"])
        w = edits.get("width", bbox["width"])
        h = edits.get("height", bbox["height"])
        
        font_name = edits.get("font_name", gemini.get("primary_font", "Roboto"))
        font_weight = edits.get("font_weight", gemini.get("font_weight", 400))
        font_size = edits.get("font_size", int(h * 0.7))
        color = edits.get("color", gemini.get("text_color", "#000000"))
        
        try:
            font_path = get_font_path(font_name, font_weight)
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        try:
            text_bbox = font.getbbox(text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            text_x = x + (w - text_w) / 2 - text_bbox[0]
            text_y = y + (h - text_h) / 2 - text_bbox[1]
            
            # Parse color
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            else:
                r, g, b = 0, 0, 0
            
            draw.text((text_x, text_y), text, fill=(r, g, b), font=font)
            
            if show_bounds:
                draw.rectangle([x, y, x + w, y + h], outline="#FF0000", width=2)
                draw.text((x, y - 15), f"Text {rid}", fill="#FF0000")
        except Exception as e:
            pass
    
    return img

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    temp_dir = Path(tempfile.gettempdir()) / "pipeline_uploads"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

def run_full_pipeline(image_path: str, progress_container):
    """Run complete pipeline"""
    try:
        progress_container.info("🔄 Stage 1: Text Detection & Layer Separation...")
        progress_bar = progress_container.progress(0.0)
        
        progress_bar.progress(0.1)
        run_dir = run_pipeline_layered(image_path, mock_layers_dir=None)
        progress_bar.progress(0.4)
        
        if not run_dir or not Path(run_dir).exists():
            raise Exception("Stage 1 failed")
        
        progress_container.info("🔄 Stage 2: Background Box Detection...")
        progress_bar.progress(0.5)
        run_box_detection_pipeline(run_dir)
        progress_bar.progress(0.7)
        
        progress_container.info("🔄 Stage 3: Final Composition...")
        progress_bar.progress(0.75)
        
        report_with_boxes = Path(run_dir) / "pipeline_report_with_boxes.json"
        report_orig = Path(run_dir) / "pipeline_report.json"
        report_file = report_with_boxes if report_with_boxes.exists() else report_orig
        
        if not report_file.exists():
            raise Exception("Pipeline report not found")
        
        with open(report_file, "r") as f:
            report = json.load(f)
        
        orig_w = report.get("original_size", {}).get("width", 1080)
        orig_h = report.get("original_size", {}).get("height", 1920)
        
        progress_bar.progress(0.8)
        base_img = composite_layers(str(run_dir), report)
        if base_img is None:
            raise Exception("Failed to composite layers")
        
        progress_bar.progress(0.85)
        base_img = draw_background_boxes(base_img, report, orig_w, orig_h, str(run_dir))
        
        progress_bar.progress(0.9)
        final_img = render_text_layer(base_img.copy(), report)
        
        out_path = Path(run_dir) / "final_composed.png"
        final_img.save(out_path)
        
        progress_bar.progress(1.0)
        progress_container.success("✅ Pipeline completed!")
        
        return run_dir, report, base_img
        
    except Exception as e:
        progress_container.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        st.error(f"```\n{traceback.format_exc()}\n```")
        return None, None, None

# Main UI
st.title("🎨 Image Editor - Upload, Process & Edit")
st.markdown("Upload an image → Process → Edit text and CTA boxes")

# Sidebar
with st.sidebar:
    st.header("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to process and edit"
    )
    
    if uploaded_file is not None:
        preview = Image.open(uploaded_file)
        st.image(preview, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🚀 Process Image", type="primary", use_container_width=True):
            with st.spinner("Saving..."):
                image_path = save_uploaded_file(uploaded_file)
            
            progress_container = st.container()
            run_dir, report, base_img = run_full_pipeline(image_path, progress_container)
            
            if run_dir and report and base_img:
                st.session_state.pipeline_data = {
                    'report': report,
                    'base_image': base_img,
                    'run_dir': str(run_dir),
                    'text_regions': get_editable_regions(report),
                    'box_regions': get_editable_boxes(report),
                    'selected_text': None,
                    'selected_box': None,
                    'edited_text': {},
                    'edited_boxes': {}
                }
                st.success("✅ Ready to edit!")
                st.rerun()

# Main editing area
if st.session_state.pipeline_data['report'] is None:
    st.info("👈 Upload an image from the sidebar to get started")
else:
    data = st.session_state.pipeline_data
    base_img = data['base_image']
    text_regions = data['text_regions']
    box_regions = data['box_regions']
    edited_text = data.get('edited_text', {})
    edited_boxes = data.get('edited_boxes', {})
    
    # Two columns: Canvas and Controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 Canvas Preview")
        
        # Render preview
        preview_img = render_editable_canvas(
            base_img, text_regions, box_regions, 
            edited_text, edited_boxes, show_bounds=True
        )
        st.image(preview_img, use_container_width=True)
        
        # Element selector
        st.subheader("📍 Select Element")
        element_type = st.radio("Type", ["Text", "CTA Box"], horizontal=True)
        
        if element_type == "Text" and text_regions:
            options = {
                f"Text {r['id']}: {r.get('gemini_analysis', {}).get('text', '')[:30]}...": r['id']
                for r in text_regions
            }
            selected = st.selectbox("Choose text", list(options.keys()))
            data['selected_text'] = options[selected]
            data['selected_box'] = None
        
        elif element_type == "CTA Box" and box_regions:
            options = {
                f"Box {b['region_id']}: {b['text'][:30]}...": b['region_id']
                for b in box_regions
            }
            selected = st.selectbox("Choose box", list(options.keys()))
            data['selected_box'] = options[selected]
            data['selected_text'] = None
    
    with col2:
        st.subheader("⚙️ Edit Properties")
        
        # Edit Text
        if data['selected_text']:
            rid = str(data['selected_text'])
            region = next((r for r in text_regions if r['id'] == rid), None)
            
            if region:
                gemini = region.get("gemini_analysis", {})
                edits = edited_text.get(rid, {})
                bbox = region["bbox"]
                
                # Text content
                current_text = edits.get("text", gemini.get("text", ""))
                new_text = st.text_area("Text", value=current_text, height=80)
                if new_text != current_text:
                    if rid not in edited_text:
                        edited_text[rid] = {}
                    edited_text[rid]["text"] = new_text
                    data['edited_text'] = edited_text
                
                # Position
                col_x, col_y = st.columns(2)
                with col_x:
                    new_x = st.number_input("X", value=edits.get("x", bbox["x"]), 
                                          min_value=0, step=5, key=f"tx_{rid}")
                with col_y:
                    new_y = st.number_input("Y", value=edits.get("y", bbox["y"]), 
                                          min_value=0, step=5, key=f"ty_{rid}")
                
                if new_x != edits.get("x", bbox["x"]) or new_y != edits.get("y", bbox["y"]):
                    if rid not in edited_text:
                        edited_text[rid] = {}
                    edited_text[rid]["x"] = int(new_x)
                    edited_text[rid]["y"] = int(new_y)
                    data['edited_text'] = edited_text
                
                # Size
                col_w, col_h = st.columns(2)
                with col_w:
                    new_w = st.number_input("Width", value=edits.get("width", bbox["width"]), 
                                          min_value=10, step=5, key=f"tw_{rid}")
                with col_h:
                    new_h = st.number_input("Height", value=edits.get("height", bbox["height"]), 
                                          min_value=10, step=5, key=f"th_{rid}")
                
                if new_w != edits.get("width", bbox["width"]) or new_h != edits.get("height", bbox["height"]):
                    if rid not in edited_text:
                        edited_text[rid] = {}
                    edited_text[rid]["width"] = int(new_w)
                    edited_text[rid]["height"] = int(new_h)
                    data['edited_text'] = edited_text
                
                # Font
                font_name = edits.get("font_name", gemini.get("primary_font", "Roboto"))
                fonts = ["Roboto", "Poppins", "Inter", "Manrope", "Plus Jakarta Sans", "Oswald"]
                new_font = st.selectbox("Font", fonts, index=fonts.index(font_name) if font_name in fonts else 0,
                                      key=f"font_{rid}")
                if new_font != font_name:
                    if rid not in edited_text:
                        edited_text[rid] = {}
                    edited_text[rid]["font_name"] = new_font
                    data['edited_text'] = edited_text
                
                # Color
                color = edits.get("color", gemini.get("text_color", "#000000"))
                new_color = st.color_picker("Color", value=color, key=f"color_{rid}")
                if new_color != color:
                    if rid not in edited_text:
                        edited_text[rid] = {}
                    edited_text[rid]["color"] = new_color
                    data['edited_text'] = edited_text
        
        # Edit Box
        elif data['selected_box']:
            bid = str(data['selected_box'])
            box = next((b for b in box_regions if str(b['region_id']) == bid), None)
            
            if box:
                edits = edited_boxes.get(bid, {})
                bbox = edits.get('bbox', box['bbox'])
                
                # Position
                col_x, col_y = st.columns(2)
                with col_x:
                    new_x = st.number_input("X", value=bbox["x"], min_value=0, step=5, key=f"bx_{bid}")
                with col_y:
                    new_y = st.number_input("Y", value=bbox["y"], min_value=0, step=5, key=f"by_{bid}")
                
                # Size
                col_w, col_h = st.columns(2)
                with col_w:
                    new_w = st.number_input("Width", value=bbox["width"], min_value=10, step=5, key=f"bw_{bid}")
                with col_h:
                    new_h = st.number_input("Height", value=bbox["height"], min_value=10, step=5, key=f"bh_{bid}")
                
                if (new_x != bbox["x"] or new_y != bbox["y"] or 
                    new_w != bbox["width"] or new_h != bbox["height"]):
                    if bid not in edited_boxes:
                        edited_boxes[bid] = {}
                    edited_boxes[bid]['bbox'] = {
                        'x': int(new_x),
                        'y': int(new_y),
                        'width': int(new_w),
                        'height': int(new_h)
                    }
                    data['edited_boxes'] = edited_boxes
                
                # Color
                color = edits.get('color', box['color'])
                new_color = st.color_picker("Box Color", value=color, key=f"bcolor_{bid}")
                if new_color != color:
                    if bid not in edited_boxes:
                        edited_boxes[bid] = {}
                    edited_boxes[bid]['color'] = new_color
                    data['edited_boxes'] = edited_boxes
        
        else:
            st.info("Select an element to edit")
        
        st.divider()
        
        # Actions
        if st.button("💾 Save Edited Image", type="primary", use_container_width=True):
            final_img = render_editable_canvas(
                base_img, text_regions, box_regions,
                edited_text, edited_boxes, show_bounds=False
            )
            output_path = Path(data['run_dir']) / "final_edited.png"
            final_img.save(output_path)
            st.success(f"✅ Saved to: {output_path}")
            
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download",
                    f.read(),
                    "final_edited.png",
                    "image/png",
                    use_container_width=True
                )
        
        if st.button("🔄 Reset All", use_container_width=True):
            data['edited_text'] = {}
            data['edited_boxes'] = {}
            st.rerun()

