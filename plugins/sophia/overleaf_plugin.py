import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict

from sophia.plugins import PluginInterface
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

class OverleafPlugin(PluginInterface):
    def name(self) -> str:
        return "overleaf"

    def register(self, registry: ToolRegistry, **kwargs) -> None:
        registry.register(
            name="overleaf_package",
            description="Package a working directory (.tex and assets) into a zip file ready for Overleaf upload.",
            parameters={
                "type": "object",
                "properties": {
                    "source_dir": {
                        "type": "string",
                        "description": "Path to the directory containing .tex files to package",
                    },
                    "output_zip": {
                        "type": "string",
                        "description": "Path where the output zip file should be saved",
                    }
                },
                "required": ["source_dir", "output_zip"]
            },
            handler=self._package_handler
        )

    def _package_handler(self, args: Dict[str, Any]) -> str:
        source_dir = args.get("source_dir")
        output_zip = args.get("output_zip")

        if not os.path.isdir(source_dir):
            return json.dumps({
                "success": False,
                "error": f"Source directory '{source_dir}' does not exist or is not a directory."
            })

        output_path = Path(output_zip)
        
        SUPPORTED_EXTS = {'.tex', '.bib', '.png', '.jpg', '.jpeg', '.pdf', '.cls', '.sty', '.bst', '.eps', '.bbl'}
        file_count = 0
        packaged_files = []

        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(source_dir):
                    # Skip hidden directories and build artifacts
                    if any(part.startswith('.') for part in Path(root).parts):
                        continue
                        
                    for file in files:
                        if file.startswith('.'):
                            continue
                            
                        file_path = Path(root) / file
                        ext = file_path.suffix.lower()
                        
                        if ext in SUPPORTED_EXTS:
                            arcname = file_path.relative_to(source_dir)
                            zipf.write(file_path, arcname)
                            packaged_files.append(str(arcname))
                            file_count += 1
                            
            instructions = (
                f"Successfully packaged {file_count} files to {output_zip}.\n\n"
                "To upload to Overleaf:\n"
                "1. Go to https://www.overleaf.com/project\n"
                "2. Click 'New Project' -> 'Upload Project'\n"
                "3. Select or drag-and-drop the generated zip file\n"
            )
                            
            return json.dumps({
                "success": True,
                "file_count": file_count,
                "files": packaged_files,
                "output_path": str(output_path),
                "instructions": instructions
            }, ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            })
