from core.report.report_engine import ReportEngine
import markdown

engine = ReportEngine()
text = "This study aims to assist pathologists in distinguishing benign from malignant breast masses using fine-needle aspirate (FNA) cell nuclei features derived from the UCI repository."
html = engine._markdown_to_html(text)
print(repr(html))
