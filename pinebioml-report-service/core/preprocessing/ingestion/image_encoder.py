import os
import base64
from typing import Dict

def encode_image_base64(path: str) -> str:
    """
    Read a PNG file and return a base64 string.
    """
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return ""


def process_images(image_dir: str) -> Dict[str, str]:
    """
    Process all PNGs in a directory, record their type,
    and build a dictionary mapping plot type to base64 string.
    """
    images_dict = {}
    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        return images_dict
        
    valid_plot_types = ['confusion_matrix', 'roc_curve', 'pca', 'pls', 'umap', 'heatmap']
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith('.png'):
            path = os.path.join(image_dir, filename)
            
            # Identify plot type
            plot_type = None
            filename_lower = filename.lower()
            
            for pt in valid_plot_types:
                # check both exact matches and without underscores
                if pt in filename_lower or pt.replace('_', '') in filename_lower.replace('_', ''):
                    plot_type = pt
                    break
            
            if plot_type:
                b64_str = encode_image_base64(path)
                if b64_str:
                    images_dict[plot_type] = b64_str
                    
    return images_dict
