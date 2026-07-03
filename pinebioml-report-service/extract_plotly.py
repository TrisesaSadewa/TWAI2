import json
import re

html_path = 'storage/media/6e66980d-df26-4c16-8727-35390e27bb00/output/feature_importance.html'
try:
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    match = re.search(r'"data":(\[.*?\]),"layout"', html)
    if match:
        data = json.loads(match.group(1))
        for trace in data:
            print(f"Name: {trace.get('name')}")
            print(f"x: {trace.get('x')}")
            print(f"y: {trace.get('y')}")
            print("-" * 20)
    else:
        print("Could not find Plotly data.")
except Exception as e:
    print(e)
