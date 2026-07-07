import os
import logging
import pypandoc
from typing import Any

logger = logging.getLogger(__name__)

# Ensure pdflatex (MiKTeX) is in PATH
miktex_path = r"C:\Users\Trisesa S\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
if miktex_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + miktex_path

# Ensure Pandoc is available
try:
    pypandoc.get_pandoc_version()
except OSError:
    logger.info("Pandoc not found. Downloading pandoc...")
    pass # pypandoc.download_pandoc()

class ExportEngine:
    """
    Handles rendering and exporting report narratives to PDF and DOCX files via Pandoc.
    """
    
    def _build_markdown(self, report_data: dict) -> str:
        """Constructs a combined markdown document from the report narrative."""
        md = []
        md.append(f"# PINEBIOML AI NARRATIVE REPORT\n\n")
        md.append(f"**Dataset Name:** {report_data.get('dataset_name', 'N/A')}  \n")
        md.append(f"**Job ID:** {report_data.get('job_id', 'N/A')}  \n")
        md.append(f"**Task Type:** {report_data.get('task_type', 'N/A').replace('_', ' ').title()}  \n")
        md.append(f"**Report ID:** {report_data.get('report_id', 'N/A')}  \n\n")
        md.append("---\n\n")
        
        narrative = report_data.get("narrative", {})
        visuals_dict = report_data.get("visuals", {})
        
        def replace_plots_with_markdown_images(text: str) -> str:
            if not text:
                return ""
            import re
            
            def sub_match(match):
                plot_keys_str = match.group(1)
                keys = [k.strip().lower() for k in plot_keys_str.split(',')]
                img_mds = []
                for k in keys:
                    clean_k = k.replace("_", "").replace(".png", "").strip()
                    match_key = next((vk for vk in visuals_dict if vk.lower().replace("_png", "").replace("_", "").replace(".png", "") == clean_k), None)
                    if match_key and visuals_dict[match_key]:
                        path = visuals_dict[match_key]
                        title = match_key.replace('_png', '').replace('.png', '').replace('_', ' ').title()
                        path_formatted = path.replace('\\', '/')
                        img_mds.append(f"![{title}]({path_formatted})")
                return "\n\n" + "\n".join(img_mds) + "\n\n"
                
            return re.sub(r'\[PLOT:\s*(.*?)\s*\]', sub_match, text, flags=re.IGNORECASE)
        
        # Expert Narrative
        md.append("# Expert Clinical Narrative\n\n")
        for sec_key in ["executive_summary", "preprocessing_and_data_quality", "findings", "visuals_analysis", "recommendations"]:
            sec_name = sec_key.replace("_", " ").replace("And", "&").title()
            md.append(f"## {sec_name}\n\n")
            content = narrative.get("expert", {}).get(sec_key, "N/A")
            
            if isinstance(content, list):
                content = "\n\n".join([str(item) for item in content])
            elif not isinstance(content, str):
                import json
                try:
                    content = json.dumps(content, indent=2)
                except Exception:
                    content = str(content) if content is not None else ""
                    
            md.append(f"{replace_plots_with_markdown_images(content)}\n\n")
            
        md.append("\\pagebreak\n\n") # Page break for PDF
        
        # Expert Narrative sections are already included above

        return "".join(md)

    def export_to_pdf(self, report_content: Any, output_pdf_path: str) -> bool:
        """
        Convert rendered HTML or report data to PDF.
        """
        try:
            from weasyprint import HTML

            logger.info(f"Rendering PDF using WeasyPrint to {output_pdf_path}...")
            if isinstance(report_content, dict):
                md_content = self._build_markdown(report_content)
                html_content = pypandoc.convert_text(md_content, 'html', format='md')
            else:
                html_content = str(report_content)

            os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
            base_url = os.path.abspath(os.path.dirname(output_pdf_path))
            HTML(string=html_content, base_url=base_url).write_pdf(output_pdf_path)
            return True
        except Exception as e:
            logger.error(f"Failed to render PDF: {e}")
            return False

    def export_to_docx(self, report_data: dict, output_docx_path: str) -> bool:
        """
        Export the markdown narrative sections to a beautifully formatted Word Document
        using an IEEE template reference docx if available.
        """
        try:
            logger.info(f"Generating Word Document via Pandoc at {output_docx_path}...")
            md_content = self._build_markdown(report_data)
            
            extra_args = []
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
            ref_docx = os.path.join(base_dir, "PineBioML", "paper_templates", "IEEE-conference-template-a4.docx")
            if os.path.exists(ref_docx):
                extra_args.append(f'--reference-doc={ref_docx}')
                
            pypandoc.convert_text(md_content, 'docx', format='md', outputfile=output_docx_path, extra_args=extra_args)
            logger.info("DOCX file saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to generate Word Document: {e}")
            return False
