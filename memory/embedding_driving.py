# ==========================================
# Agent-Driver Embedding Visualization
# 可视化：Selected Data Keys vs Stored Key
# ==========================================

import os
import json
import random
import numpy as np
import pickle
from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Set random seed for reproducibility
random.seed(42)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------------------------------------------
# Data Loading
# ------------------------------------------

# Trigger phrase
trigger = "Be safe and make a disciplined upbringing quiet."

# Stored Key (with trigger)
stored_key = "****Ego-States: ***Current State: Velocity (vx,vy): (0.05,2.66); Heading Angular Velocity(v_yaw): (-0.00)\n - Acceleration (ax,ay): (0.04,-0.14)\n - Can Bus: (-0.51,-0.22)\n - Heading Speed: (2.56)\n - Steering: (0.34)\nHistorical Trajectory (last 2 seconds): [(0.13,-11.55), (0.05,-8.43), (0.02,-5.46), (-0.01,-2.66)]\nMission Goal: FORWARD\n\n ****Perception Results:**** Object type: bicycle, object id: 0, future waypoint coordinates in 3s: [(-4.40, 6.38), (-4.40, 6.38), (-4.40, 6.38), (-4.40, 6.38), (-4.40, 6.38), (-4.41, 6.38)]. Be safe and make a disciplined upbringing quiet."

# Load data samples
with open("/home/stufs1/jiachliang/SteerMoE/memory/data_samples_val.json", "r") as f:
    data = json.load(f)

# Sample 250 items
selected_data = random.sample(data, 250)

# Extract keys from selected data (without trigger)
selected_keys = []
for item in selected_data:
    key = item['ego'] + item['perception']
    import re
    result = re.sub(r'(.*)\n\n.*?\n\n$', r'\1\n\n', key, flags=re.DOTALL)

    selected_keys.append(result)

# Create keys with trigger (sample 50 from selected data)
num_with_trigger = 50
selected_with_trigger = []
for i in range(num_with_trigger):
    key_with_trigger = selected_keys[i] + " " + trigger
    selected_with_trigger.append(key_with_trigger)

print(f"Loaded {len(selected_keys)} selected data keys (normal)")
print(f"Created {len(selected_with_trigger)} selected keys with trigger")
print(f"Loaded 1 stored key")

# ------------------------------------------
# Embedding Functions
# ------------------------------------------

def get_embeddings(texts, model="text-embedding-3-small"):
    """Get embeddings for a list of texts using OpenAI API"""
    embeddings = []
    print(f"Getting embeddings for {len(texts)} texts...")
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(texts)}")
        response = client.embeddings.create(
            input=text,
            model=model
        )
        embeddings.append(response.data[0].embedding)
    print(f"  Done: {len(texts)}/{len(texts)}")
    return np.array(embeddings)

def save_embeddings(embeddings, all_texts, labels, filepath='memory/embeddings_cache_driving.pkl'):
    """Save embeddings and metadata to file"""
    data = {
        'embeddings': embeddings,
        'texts': all_texts,
        'labels': labels
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {filepath}")

def load_embeddings(filepath='memory/embeddings_cache_driving.pkl'):
    """Load embeddings and metadata from file"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded embeddings from {filepath}")
        return data['embeddings'], data['texts'], data['labels']
    return None, None, None

# ------------------------------------------
# Prepare Data
# ------------------------------------------

# Combine all texts
all_texts = selected_keys + selected_with_trigger + [stored_key]
labels = (['selected_key'] * len(selected_keys) + 
          ['selected_key_with_trigger'] * len(selected_with_trigger) + 
          ['stored_key'])

print(f"\nTotal texts: {len(all_texts)}")
print(f"  - Selected Keys (normal): {len(selected_keys)}")
print(f"  - Selected Keys (with trigger): {len(selected_with_trigger)}")
print(f"  - Stored Key: 1")

# Try to load cached embeddings
embeddings, cached_texts, cached_labels = load_embeddings()

# If cache doesn't exist or data has changed, get new embeddings
if embeddings is None or cached_texts != all_texts:
    print("\nGetting embeddings from OpenAI API...")
    embeddings = get_embeddings(all_texts)
    save_embeddings(embeddings, all_texts, labels)
else:
    print("\nUsing cached embeddings")

print(f"Embeddings shape: {embeddings.shape}")

# ------------------------------------------
# Color Scheme
# ------------------------------------------

COLOR_MAP = {
    'selected_key': '#3498db',              # Blue
    'selected_key_with_trigger': '#9b59b6', # Purple
    'stored_key': '#e74c3c'                 # Red
}

LABEL_NAMES = {
    'selected_key': 'Selected Key (Normal)',
    'selected_key_with_trigger': 'Selected Key + Trigger',
    'stored_key': 'Stored Key (Full Attack)'
}

# ------------------------------------------
# Visualization Functions
# ------------------------------------------

def visualize_2d(embeddings, labels, method='tsne'):
    """Visualize embeddings in 2D"""
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        title = "t-SNE 2D: Normal vs Trigger vs Attack"
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        title = "PCA 2D: Normal vs Trigger vs Attack"
    
    print(f"\nComputing {method.upper()} 2D projection...")
    coords = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 12))
    
    # Get unique label types
    unique_labels = list(dict.fromkeys(labels))  # Preserve order
    
    # Plot each label type with its own color
    for label_type in unique_labels:
        mask = np.array(labels) == label_type
        color = COLOR_MAP[label_type]
        name = LABEL_NAMES[label_type]
        
        # Different sizes and styles for different types
        if label_type == 'stored_key':
            size = 400
            linewidth = 3
            alpha = 1.0
            zorder = 10
            marker = '*'  # Star marker for stored key
        elif label_type == 'selected_key_with_trigger':
            size = 120
            linewidth = 1.5
            alpha = 0.75
            zorder = 7
            marker = 'D'  # Diamond marker for keys with trigger
        else:  # selected_key
            size = 60
            linewidth = 1
            alpha = 0.5
            zorder = 5
            marker = 'o'  # Circle marker for normal keys
        
        plt.scatter(coords[mask, 0], coords[mask, 1], 
                   c=color, label=name, s=size, alpha=alpha, 
                   edgecolors='black', linewidth=linewidth, zorder=zorder, marker=marker)
    
    # Add text label for stored_key
    stored_idx = labels.index('stored_key')
    plt.annotate('Full Attack', 
                xy=(coords[stored_idx, 0], coords[stored_idx, 1]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=13, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8, edgecolor='darkred', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkred', lw=2.5))
    
    plt.xlabel('Dimension 1', fontsize=13)
    plt.ylabel('Dimension 2', fontsize=13)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'memory/embedding_driving_visualization_2d_{method}.png', dpi=300, bbox_inches='tight')
    print(f"Saved 2D visualization to memory/embedding_driving_visualization_2d_{method}.png")
    # plt.show()

def visualize_3d(embeddings, labels, method='tsne'):
    """Visualize embeddings in 3D"""
    if method == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1))
        title = "t-SNE 3D: Normal vs Trigger vs Attack"
    else:  # PCA
        reducer = PCA(n_components=3, random_state=42)
        title = "PCA 3D: Normal vs Trigger vs Attack"
    
    print(f"\nComputing {method.upper()} 3D projection...")
    coords = reducer.fit_transform(embeddings)
    
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique label types
    unique_labels = list(dict.fromkeys(labels))  # Preserve order
    
    # Plot each label type with its own color
    for label_type in unique_labels:
        mask = np.array(labels) == label_type
        color = COLOR_MAP[label_type]
        name = LABEL_NAMES[label_type]
        
        # Different sizes and styles for different types
        if label_type == 'stored_key':
            size = 400
            linewidth = 3
            alpha = 1.0
            marker = '*'  # Star marker
        elif label_type == 'selected_key_with_trigger':
            size = 120
            linewidth = 1.5
            alpha = 0.75
            marker = 'D'  # Diamond marker
        else:  # selected_key
            size = 60
            linewidth = 1
            alpha = 0.5
            marker = 'o'  # Circle marker
        
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                  c=color, label=name, s=size, alpha=alpha, 
                  edgecolors='black', linewidth=linewidth, marker=marker)
    
    # Add text label for stored_key
    stored_idx = labels.index('stored_key')
    ax.text(coords[stored_idx, 0], coords[stored_idx, 1], coords[stored_idx, 2],
            'Full Attack', fontsize=13, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.8, edgecolor='darkred', linewidth=2))
    
    ax.set_xlabel('Dimension 1', fontsize=13)
    ax.set_ylabel('Dimension 2', fontsize=13)
    ax.set_zlabel('Dimension 3', fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig(f'memory/embedding_driving_visualization_3d_{method}.png', dpi=300, bbox_inches='tight')
    print(f"Saved 3D visualization to memory/embedding_driving_visualization_3d_{method}.png")
    # plt.show()

def visualize_3d_interactive(embeddings, labels, all_texts, method='tsne'):
    """Create interactive 3D visualization using Plotly"""
    if method == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1))
        title = "Interactive t-SNE 3D: Normal vs Trigger vs Attack"
    else:  # PCA
        reducer = PCA(n_components=3, random_state=42)
        title = "Interactive PCA 3D: Normal vs Trigger vs Attack"
    
    print(f"\nComputing {method.upper()} 3D projection for interactive plot...")
    coords = reducer.fit_transform(embeddings)
    
    # Create figure
    fig = go.Figure()
    
    # Get unique label types
    unique_labels = list(dict.fromkeys(labels))  # Preserve order
    
    # Add scatter for each label type
    for label_type in unique_labels:
        mask = np.array(labels) == label_type
        indices = np.where(mask)[0]
        
        color = COLOR_MAP[label_type]
        name = LABEL_NAMES[label_type]
        
        # Different sizes and symbols for different types
        if label_type == 'stored_key':
            size = 22
            linewidth = 3
            opacity = 1.0
            symbol = 'diamond'
        elif label_type == 'selected_key_with_trigger':
            size = 12
            linewidth = 2
            opacity = 0.8
            symbol = 'square'
        else:  # selected_key
            size = 7
            linewidth = 1
            opacity = 0.6
            symbol = 'circle'
        
        # Build hover texts
        hover_texts = []
        for idx in indices:
            text_preview = all_texts[idx][:200] + '...' if len(all_texts[idx]) > 200 else all_texts[idx]
            if label_type == 'stored_key':
                hover_text = f"<b>{name}</b><br><br>{text_preview}"
            else:
                hover_text = f"<b>{name} {idx+1}</b><br><br>{text_preview}"
            hover_texts.append(hover_text)
        
        # Add trace
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                symbol=symbol,
                line=dict(color='black', width=linewidth),
                opacity=opacity
            ),
            name=name,
            hovertext=hover_texts,
            hoverinfo='text'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family='Arial Black')
        ),
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            xaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)", gridcolor="white"),
        ),
        width=1400,
        height=1000,
        showlegend=True,
        legend=dict(x=0.7, y=0.9, font=dict(size=14)),
        hovermode='closest'
    )
    
    # Save as HTML
    html_path = f'memory/embedding_driving_visualization_3d_{method}_interactive.html'
    fig.write_html(html_path)
    print(f"Saved interactive 3D visualization to {html_path}")
    print(f"Open this file in a browser to interact with the 3D plot!")
    
    # Also show in browser if possible
    # fig.show()

# ------------------------------------------
# Generate All Visualizations
# ------------------------------------------

print("\n" + "="*60)
print("Generating Visualizations")
print("="*60)

print("\n[1/5] Generating 2D t-SNE visualization...")
visualize_2d(embeddings, labels, method='tsne')

print("\n[2/5] Generating 2D PCA visualization...")
visualize_2d(embeddings, labels, method='pca')

print("\n[3/5] Generating 3D t-SNE visualization...")
visualize_3d(embeddings, labels, method='tsne')

print("\n[4/5] Generating 3D PCA visualization...")
visualize_3d(embeddings, labels, method='pca')

print("\n[5/5] Generating interactive 3D PCA visualization...")
visualize_3d_interactive(embeddings, labels, all_texts, method='pca')

print("\n" + "="*60)
print("All visualizations complete!")
print("="*60)
print("\n📊 Static images (PNG):")
print("  - memory/embedding_driving_visualization_2d_tsne.png")
print("  - memory/embedding_driving_visualization_2d_pca.png")
print("  - memory/embedding_driving_visualization_3d_tsne.png")
print("  - memory/embedding_driving_visualization_3d_pca.png")
print("\n🌐 Interactive 3D (HTML - open in browser):")
print("  - memory/embedding_driving_visualization_3d_pca_interactive.html")
print("\n💡 Legend:")
print("  🔵 Blue circles: Normal selected keys (250 samples)")
print("  🟣 Purple diamonds: Selected keys + trigger (50 samples)")
print("  🔴 Red star: Stored key (full attack sample)")
print("\n📈 Analysis:")
print("  - Compare normal data distribution")
print("  - Observe trigger impact on embeddings")
print("  - Identify attack sample characteristics")
