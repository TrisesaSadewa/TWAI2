import re

file_path = r'c:\Users\Trisesa S\Documents\TRS\ITS\IIPP\TWAI2\pinebioml-report-service\core\report\report_engine.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

pattern_url = r'return f"/api/artifacts/\{report_id\}/\{quote\(filename\)\}"'
replacement_url = r'return f"/media/{report_id}/output/{quote(filename)}"'

content = re.sub(pattern_url, replacement_url, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Artifact URL replaced successfully")

