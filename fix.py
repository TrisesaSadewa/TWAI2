import re

file_path = r'c:\Users\Trisesa S\Documents\TRS\ITS\IIPP\TWAI2\pinebioml-report-service\core\report\report_engine.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

pattern_md = r'(def _markdown_to_html\(self, text\) -> str:.*?)(?=\s+def _replace_plots_with_html)'
replacement_md = r'''def _markdown_to_html(self, text) -> str:
        \"\"\"Converts basic markdown to HTML for server-side rendering (e.g. for PDF export).\"\"\"
        from core.security import sanitize_html_content
        import markdown
        if not text:
            return ""
            
        if isinstance(text, list):
            text = "\n\n".join([str(t) for t in text])
        elif not isinstance(text, str):
            text = str(text)
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Sanitize dangerous HTML tags/event handlers from LLM output
        text = sanitize_html_content(text)
        
        html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'nl2br'])
        return html'''

content = re.sub(pattern_md, replacement_md, content, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Markdown to html replaced successfully")

