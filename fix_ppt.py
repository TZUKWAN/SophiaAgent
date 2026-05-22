import re

with open('sophia/research/ppt_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure python-pptx is available or imported if needed
if "import pptx" not in content and "from pptx" not in content:
    content = "try:\n    from pptx import Presentation\n    from pptx.util import Inches, Pt\n    PPTX_AVAILABLE = True\nexcept ImportError:\n    PPTX_AVAILABLE = False\n" + content

# Add PPTX generating class
pptx_class_code = """
class PPTXSlideRenderer:
    \"\"\"Render slides as a real PPTX file.\"\"\"
    
    def render(self, args: dict) -> Dict[str, Any]:
        \"\"\"Render slides to a real PPTX file using python-pptx.
        
        Args:
            slides (List[Dict]): Slides from generate_structure
            title (str): Presentation title
            output_path (str): File path to save
            image_paths (Dict[int, str]): Map of slide index to image path (e.g. {1: "/tmp/plot.png"})
        \"\"\"
        if not PPTX_AVAILABLE:
            return {"error": "python-pptx package is not installed. Add it to dependencies."}
            
        slides_data = args.get("slides", [])
        title = args.get("title", "Presentation")
        output_path = args.get("output_path", "presentation.pptx")
        image_paths = args.get("image_paths", {})
        
        # Make int keys out of str keys if any
        images = {}
        for k, v in image_paths.items():
            try:
                images[int(k)] = v
            except ValueError:
                pass
                
        prs = Presentation()
        
        # Title slide layout
        title_slide_layout = prs.slide_layouts[0]
        # Bullet slide layout
        bullet_slide_layout = prs.slide_layouts[1]
        
        for idx, slide_data in enumerate(slides_data):
            slide_type = slide_data.get("type", "content")
            slide_title_text = slide_data.get("title", f"Slide {idx+1}")
            bullets = slide_data.get("content_bullets", [])
            notes = slide_data.get("speaker_notes", "")
            
            if slide_type == "title" or slide_type == "final":
                slide = prs.slides.add_slide(title_slide_layout)
                title_shape = slide.shapes.title
                subtitle_shape = slide.placeholders[1]
                
                title_shape.text = slide_title_text
                
                if bullets:
                    subtitle_shape.text = "\n".join(bullets)
                else:
                    subtitle_shape.text = title if slide_type == "title" else "Thank You"
            else:
                slide = prs.slides.add_slide(bullet_slide_layout)
                shapes = slide.shapes
                title_shape = shapes.title
                body_shape = shapes.placeholders[1]
                
                title_shape.text = slide_title_text
                
                tf = body_shape.text_frame
                for i, bullet in enumerate(bullets):
                    if i == 0:
                        tf.text = bullet
                    else:
                        p = tf.add_paragraph()
                        p.text = bullet
                        p.level = 0
            
            # Add images if provided for this slide
            if idx in images and images[idx]:
                import os
                if os.path.exists(images[idx]):
                    # Add picture (left=1 inch, top=2 inches, width=5 inches)
                    # Adjust parameters dynamically based on slide
                    try:
                        slide.shapes.add_picture(images[idx], Inches(2), Inches(2), width=Inches(6))
                    except Exception as e:
                        print(f"Failed to add image to slide {idx}: {e}")
                        
            # Add speaker notes
            if hasattr(slide, "notes_slide") and notes:
                notes_slide = slide.notes_slide
                text_frame = notes_slide.notes_text_frame
                text_frame.text = notes
                
        prs.save(output_path)
        
        return {
            "status": "success",
            "format": "pptx",
            "path": output_path,
            "slide_count": len(slides_data),
            "images_embedded": len(images)
        }
"""

content += "\n" + pptx_class_code

with open('sophia/research/ppt_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)
