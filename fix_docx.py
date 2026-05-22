import re

with open('sophia/exporters/docx_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add progress_callback parameter to export_paper
pattern1 = """    def export_paper(
        self,
        doc: Dict[str, Any],
        output_path: str,
        citation_style: str = "apa7",
        include_results: bool = True,
        result_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:"""

replacement1 = """    def export_paper(
        self,
        doc: Dict[str, Any],
        output_path: str,
        citation_style: str = "apa7",
        include_results: bool = True,
        result_ids: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:"""

# Add calls to progress_callback
pattern2 = """        document = Document()
        self._setup_styles(document)
        self._setup_page(document)

        warnings: List[str] = []

        # Title page
        self._render_title_page(document, doc)

        # Abstract
        if doc.get("abstract"):
            self._render_abstract(document, doc["abstract"], doc.get("keywords", []))

        # Sections
        sections = doc.get("sections", {})
        results_data: List[Dict] = []

        if include_results and result_ids is None:
            result_ids = self._extract_result_ids_from_doc(doc)

        if include_results and result_ids and self.store:
            for rid in result_ids:
                try:
                    data = self.store.get(rid)
                    meta = self.store.get_metadata(rid)
                    results_data.append({"result_id": rid, "data": data, "meta": meta})
                except Exception as exc:
                    warnings.append(f"result_id {rid}: {exc}")

        for key in sorted(sections.keys(), key=lambda x: int(x)):
            section = sections[key]
            title = section.get("title", "")
            content = section.get("content", "")
            self._render_section(document, title, content, results_data, citation_style)

        # References
        refs = doc.get("references", [])
        if refs:
            self._render_references(document, refs, citation_style)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        document.save(output_path)"""

replacement2 = """        if progress_callback: progress_callback(("init", 0.05))
        document = Document()
        self._setup_styles(document)
        self._setup_page(document)

        warnings: List[str] = []

        if progress_callback: progress_callback(("rendering_title", 0.1))
        # Title page
        self._render_title_page(document, doc)

        # Abstract
        if progress_callback: progress_callback(("rendering_abstract", 0.15))
        if doc.get("abstract"):
            self._render_abstract(document, doc["abstract"], doc.get("keywords", []))

        # Sections
        sections = doc.get("sections", {})
        results_data: List[Dict] = []

        if include_results and result_ids is None:
            result_ids = self._extract_result_ids_from_doc(doc)

        if progress_callback: progress_callback(("preparing_results", 0.2))
        if include_results and result_ids and self.store:
            for rid in result_ids:
                try:
                    data = self.store.get(rid)
                    meta = self.store.get_metadata(rid)
                    results_data.append({"result_id": rid, "data": data, "meta": meta})
                except Exception as exc:
                    warnings.append(f"result_id {rid}: {exc}")

        sorted_keys = sorted(sections.keys(), key=lambda x: int(x))
        total_sections = len(sorted_keys)
        for i, key in enumerate(sorted_keys):
            if progress_callback: 
                progress = 0.2 + 0.6 * (i / max(1, total_sections))
                progress_callback((f"rendering_section_{key}", progress))
            section = sections[key]
            title = section.get("title", "")
            content = section.get("content", "")
            self._render_section(document, title, content, results_data, citation_style)

        # References
        if progress_callback: progress_callback(("rendering_references", 0.85))
        refs = doc.get("references", [])
        if refs:
            self._render_references(document, refs, citation_style)

        # Save
        if progress_callback: progress_callback(("saving", 0.95))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        document.save(output_path)
        if progress_callback: progress_callback(("done", 1.0))"""

content = content.replace(pattern1, replacement1)
content = content.replace(pattern2, replacement2)

with open('sophia/exporters/docx_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

