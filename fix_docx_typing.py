import re

with open('sophia/exporters/docx_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure Callable is imported if not already there
if "from typing import" in content and "Callable" not in content.split("from typing import")[1].split("\n")[0]:
    content = content.replace("from typing import ", "from typing import Callable, ")

# Change callable to Callable
content = content.replace("progress_callback: Optional[callable] = None", "progress_callback: Optional[Callable] = None")

with open('sophia/exporters/docx_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
