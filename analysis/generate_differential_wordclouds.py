#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate differential word clouds with different colors for significantly increased and decreased words after concretization
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import re
from collections import Counter
import os

# Set font for matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Read data
print("Reading data...")
df = pd.read_csv('./data/jailbreak-dataset-analysis.csv')  # Update this path according to your setup

print(f"Number of rows: {len(df):,}")

# Define stopwords (common English stopwords)
stopwords = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'up', 'about', 'into', 'through', 'during', 'including', 'against', 'among',
    'throughout', 'towards', 'upon', 'concerning', 'to', 'of', 'in', 'for',
    'on', 'at', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
    'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 
    'some', 'such','your','their','them','there','like','without',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', 'should', 'now', 'one', 'two', 'three', 'maybe', 'what', 'okay',
    'also', 'another', 'way', 'ways', 'use', 'used', 'using', 'get', 'got', 'make', 'made',
    'take', 'took', 'come', 'came', 'go', 'went', 'see', 'saw', 'know', 'knew', 'think',
    'thought', 'say', 'said', 'tell', 'told', 'ask', 'asked', 'want', 'wanted', 'need',
    'needed', 'give', 'gave', 'show', 'showed', 'work',
    'worked', 'call', 'called', 'look', 'looked', 'seem', 'seemed'
])

def clean_text(text):
    """Clean text and extract words"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Extract words
    words = text.split()
    # Filter stopwords and short words
    words = [w for w in words if len(w) > 3 and w not in stopwords]
    return words

def get_word_frequencies(text_list):
    """Get word frequency dictionary"""
    all_words = []
    for words in text_list:
        all_words.extend(words)
    return Counter(all_words)

def calculate_word_changes(original_texts, concrete_texts):
    """
    Calculate word changes before and after concretization
    Returns: (increased_words, decreased_words, word_changes_dict)
    - increased_words: dictionary of significantly increased words {word: change_info}
    - decreased_words: dictionary of significantly decreased words {word: change_info}
    - word_changes_dict: dictionary of all word changes {word: change_info}
    """
    # Get word frequencies
    original_freq = get_word_frequencies(original_texts)
    concrete_freq = get_word_frequencies(concrete_texts)
    
    # Get all words
    all_words = set(original_freq.keys()) | set(concrete_freq.keys())
    
    word_changes = {}
    increased_words = {}
    decreased_words = {}
    
    # Calculate total word count (for normalization)
    original_total = sum(original_freq.values())
    concrete_total = sum(concrete_freq.values())
    
    for word in all_words:
        orig_count = original_freq.get(word, 0)
        conc_count = concrete_freq.get(word, 0)
        
        # Calculate normalized frequency
        orig_freq = orig_count / original_total if original_total > 0 else 0
        conc_freq = conc_count / concrete_total if concrete_total > 0 else 0
        
        # Calculate change ratio (relative change)
        if orig_freq > 0:
            change_ratio = (conc_freq - orig_freq) / orig_freq
        elif conc_freq > 0:
            change_ratio = 10.0  # New words, set to large value
        else:
            change_ratio = 0.0
        
        # Calculate absolute change
        abs_change = conc_freq - orig_freq
        
        word_changes[word] = {
            'original_freq': orig_freq,
            'concrete_freq': conc_freq,
            'original_count': orig_count,
            'concrete_count': conc_count,
            'change_ratio': change_ratio,
            'abs_change': abs_change
        }
        
        # Determine significant changes (based on change ratio and absolute change)
        # Significantly increased: change_ratio > 0.5 and abs_change > 0.0001, or new words
        # Significantly decreased: change_ratio < -0.3 and original_freq > 0.0001
        if (change_ratio > 0.5 and abs_change > 0.0001) or (orig_count == 0 and conc_count > 0):
            increased_words[word] = {
                'change_ratio': change_ratio,
                'abs_change': abs_change,
                'concrete_count': conc_count
            }
        elif change_ratio < -0.3 and orig_freq > 0.0001:
            decreased_words[word] = {
                'change_ratio': change_ratio,
                'abs_change': abs_change,
                'original_count': orig_count
            }
    
    return increased_words, decreased_words, word_changes

def generate_differential_wordcloud(increased_words, decreased_words, title, filename, max_words=150):
    """
    Generate differential word cloud with different colors for increased and decreased words
    - Increased words: red colors
    - Decreased words: green colors
    """
    if not increased_words and not decreased_words:
        print(f"Warning: {title} has insufficient changing words")
        return
    
    # Prepare word frequency dictionary (for WordCloud)
    word_freq_dict = {}
    
    # Add increased words (use concrete frequency)
    for word, info in increased_words.items():
        # Use concrete word frequency, weighted by change ratio
        word_freq_dict[word] = info['concrete_count'] * (1 + abs(info['change_ratio']))
    
    # Add decreased words (use original frequency)
    for word, info in decreased_words.items():
        # Use original word frequency, weighted by change ratio
        word_freq_dict[word] = info['original_count'] * (1 + abs(info['change_ratio']))
    
    if not word_freq_dict:
        print(f"Warning: {title} has no valid word frequency data")
        return
    
    try:
        # Create custom color function
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            """Assign colors based on whether word is increased or decreased"""
            if word in increased_words:
                # Increased words: light red tones (from light red to medium red)
                # Adjust color depth based on change ratio
                change_ratio = increased_words[word]['change_ratio']
                if change_ratio > 5:
                    return (240, 150, 150)  # Medium-light red: new words
                elif change_ratio > 2:
                    return (255, 170, 170)  # Light red: major increase
                else:
                    return (255, 200, 200)  # Very light red: moderate increase
            elif word in decreased_words:
                # Decreased words: green tones (from light green to dark green)
                change_ratio = abs(decreased_words[word]['change_ratio'])
                if change_ratio > 0.5:
                    return (0, 150, 0)  # Dark green: major decrease
                else:
                    return (100, 200, 100)  # Light green: moderate decrease
            else:
                return (128, 128, 128)  # Gray: other cases (should not appear in theory)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=1600,
            height=1200,
            background_color='white',
            max_words=max_words,
            relative_scaling=0.5,
            random_state=42,
            color_func=color_func,
            collocations=False
        ).generate_from_frequencies(word_freq_dict)
        
        # Create figure without legend (legend will be in combined figure)
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        # Add title at the top of the figure
        plt.title(title, fontsize=52, pad=20, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {filename}")
        
        # Print statistics
        print(f"  - Significantly increased words: {len(increased_words)}")
        print(f"  - Significantly decreased words: {len(decreased_words)}")
        
    except Exception as e:
        print(f"Error generating {title} differential word cloud: {e}")
        import traceback
        traceback.print_exc()

# Create output directory
output_dir = './output/wordclouds_differential'  # Update this path according to your setup
os.makedirs(output_dir, exist_ok=True)

print("\nProcessing text data...")

# Process columns
columns_to_process = {
    'question': 'Question',
    'concrete': 'Concrete',
    'reasoning': 'Reasoning',
    'concrete_reasoning': 'Concrete Reasoning',
    'response': 'Response',
    'concrete_response': 'Concrete Response'
}

all_texts_processed = {}
for col, title in columns_to_process.items():
    print(f"Processing {title}...")
    texts = df[col].dropna().astype(str).tolist()
    cleaned_texts = [clean_text(t) for t in texts]
    all_texts_processed[col] = cleaned_texts

# Define comparison pairs with titles (simplified for combined figure)
comparison_pairs = [
    ('question', 'concrete', '(a) Prompt'),
    ('reasoning', 'concrete_reasoning', '(b) Reasoning'),
    ('response', 'concrete_response', '(c) Response')
]

print("\nGenerating differential word clouds...")
print("=" * 80)

for orig_col, conc_col, pair_name in comparison_pairs:
    if orig_col not in all_texts_processed or conc_col not in all_texts_processed:
        print(f"Warning: Skipping {pair_name}, missing required columns")
        continue
    
    print(f"\nAnalyzing {pair_name}...")
    original_texts = all_texts_processed[orig_col]
    concrete_texts = all_texts_processed[conc_col]
    
    # Calculate word changes
    increased_words, decreased_words, word_changes = calculate_word_changes(
        original_texts, concrete_texts
    )
    
    # Generate differential word cloud
    filename = os.path.join(output_dir, f'{orig_col}_vs_{conc_col}_differential_wordcloud.png')
    generate_differential_wordcloud(
        increased_words,
        decreased_words,
        pair_name,
        filename,
        max_words=200
    )
    
    # Save detailed change statistics
    stats_filename = os.path.join(output_dir, f'{orig_col}_vs_{conc_col}_word_changes.txt')
    with open(stats_filename, 'w', encoding='utf-8') as f:
        f.write(f"Word Change Statistics: {pair_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Significantly Increased Words (Total: {len(increased_words)}):\n")
        f.write("-" * 80 + "\n")
        sorted_increased = sorted(increased_words.items(), 
                                 key=lambda x: x[1]['change_ratio'], reverse=True)
        for word, info in sorted_increased[:50]:  # Show top 50
            f.write(f"{word:20s} | Change Ratio: {info['change_ratio']:7.2f} | "
                   f"Absolute Change: {info['abs_change']:8.6f} | "
                   f"Concrete Count: {info['concrete_count']}\n")
        
        f.write(f"\nSignificantly Decreased Words (Total: {len(decreased_words)}):\n")
        f.write("-" * 80 + "\n")
        sorted_decreased = sorted(decreased_words.items(), 
                                 key=lambda x: x[1]['change_ratio'])
        for word, info in sorted_decreased[:50]:  # Show top 50
            f.write(f"{word:20s} | Change Ratio: {info['change_ratio']:7.2f} | "
                   f"Absolute Change: {info['abs_change']:8.6f} | "
                   f"Original Count: {info['original_count']}\n")
    
    print(f"  - Saved detailed statistics: {stats_filename}")

print("\n" + "=" * 80)
print("All differential word clouds generated successfully!")
print(f"Output directory: {output_dir}")
print("=" * 80)

# List generated files
print("\nGenerated files:")
files = os.listdir(output_dir)
for f in sorted(files):
    if f.endswith('.png'):
        filepath = os.path.join(output_dir, f)
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"  - {f} ({size:.1f} KB)")
    elif f.endswith('.txt'):
        print(f"  - {f}")
