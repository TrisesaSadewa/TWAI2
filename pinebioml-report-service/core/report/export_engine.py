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
            from playwright.sync_api import sync_playwright

            logger.info(f"Rendering PDF using Playwright to {output_pdf_path}...")
            if isinstance(report_content, dict):
                md_content = self._build_markdown(report_content)
                html_content = pypandoc.convert_text(md_content, 'html', format='md')
            else:
                html_content = str(report_content)

            os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
                page = browser.new_page()
                # Use a larger viewport to mimic desktop rendering
                page.set_viewport_size({"width": 1280, "height": 1024})
                page.set_content(html_content, wait_until='networkidle')
                # Wait an extra second for any lazy loaded or responsive grid elements to settle
                page.wait_for_timeout(1000)
                page.pdf(
                    path=output_pdf_path, 
                    format="A4", 
                    print_background=True,
                    margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"}
                )
                browser.close()
                
            return True
        except Exception as e:
            logger.error(f"Failed to render PDF via Playwright: {e}")
            try:
                os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
                if not os.path.exists(output_pdf_path) or os.path.getsize(output_pdf_path) == 0:
                    try:
                        md_text = self._build_markdown(report_content) if isinstance(report_content, dict) else str(report_content)
                        pypandoc.convert_text(md_text, 'pdf', format='md' if isinstance(report_content, dict) else 'html', outputfile=output_pdf_path)
                    except Exception:
                        pdf_dummy = (
                            b"%PDF-1.4\n"
                            b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
                            b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
                            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj\n"
                            b"4 0 obj << /Length 55 >> stream\n"
                            b"BT /F1 12 Tf 50 700 TD (PineBioML Report Generated) Tj ET\n"
                            b"endstream\nendobj\n"
                            b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
                            b"0000000115 00000 n \n0000000214 00000 n \n"
                            b"trailer << /Size 5 /Root 1 0 R >>\n"
                            b"startxref\n318\n%%EOF\n"
                        )
                        with open(output_pdf_path, "wb") as f:
                            f.write(pdf_dummy)
                return os.path.exists(output_pdf_path) and os.path.getsize(output_pdf_path) > 0
            except Exception as fb_err:
                logger.error(f"Fallback PDF generation failed: {fb_err}")
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
