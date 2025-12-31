#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine three differential word cloud images into a single figure
"""

import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import os

# Set font for matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Input image paths - please update this path according to your setup
input_dir = './output/wordclouds_differential'
images = [
    ('question_vs_concrete_differential_wordcloud.png', ''),
    ('reasoning_vs_concrete_reasoning_differential_wordcloud.png', ''),
    ('response_vs_concrete_response_differential_wordcloud.png', '')
]

# Output paths
output_path_png = os.path.join(input_dir, 'combined_differential_wordclouds.png')
output_path_pdf = os.path.join(input_dir, 'combined_differential_wordclouds.pdf')

print("Loading images...")
# Create figure - just combine images without any annotations
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for idx, (img_name, label) in enumerate(images):
    img_path = os.path.join(input_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found!")
        continue
    
    # Load image
    img = mpimg.imread(img_path)
    
    # Display image directly without any modifications
    axes[idx].imshow(img)
    axes[idx].axis('off')

# No legend, no caption, no annotations - just the images
plt.tight_layout(pad=0)
# Save PNG
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
# Save PDF
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', pad_inches=0)
plt.close()

print(f"Combined image saved to: {output_path_png}")
file_size_png = os.path.getsize(output_path_png) / (1024 * 1024)  # MB
print(f"PNG file size: {file_size_png:.2f} MB")

print(f"Combined PDF saved to: {output_path_pdf}")
file_size_pdf = os.path.getsize(output_path_pdf) / (1024 * 1024)  # MB
print(f"PDF file size: {file_size_pdf:.2f} MB")

