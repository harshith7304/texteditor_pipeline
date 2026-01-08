import sys
import os
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.append(str(project_root))

from pipeline_v4.run_pipeline_layered_v4 import run_pipeline_layered
from pipeline_v4.run_pipeline_v4 import run_pipeline_v4

def main():
    # Use the absolute path or relative from project root which is cwd
    image_rel_path = r"image/IMAGE_CTA_BOX/Sparkling Family Joy.png"
    image_path = Path(image_rel_path)
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path.absolute()}")
        return

    print(f"Starting Manual Pipeline Run for: {image_path}")
    
    try:
        # Step 1: Layered Pipeline (CRAFT + Gemini + Qwen + Cleaning)
        print("\n--- STAGE 1: Layered Pipeline ---")
        run_dir = run_pipeline_layered(str(image_path))
        print(f"Layered Pipeline Output Directory: {run_dir}")

        # Step 2 & 3: Box Detection & Text Rendering (Combined Orchestrator)
        print("\n--- STAGE 2 & 3: Box Detection & Rendering ---")
        run_pipeline_v4(run_dir)
        
        print("\n✅ Full Pipeline Run Complete!")

    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
