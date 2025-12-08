
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Configure premium style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'input': '#E3F2FD', # Light Blue
    'layer': '#F5F5F5', # Grey
    'conv': '#FFF3E0',  # Orange tint
    'pool': '#E8F5E9',  # Green tint
    'output': '#E3F2FD',
    'special': '#FFF9C4', # Yellow tint
    'border_input': '#1565C0',
    'border_layer': '#616161',
    'border_conv': '#EF6C00',
    'border_pool': '#2E7D32',
    'border_output': '#1565C0',
    'border_special': '#FBC02D'
}

def draw_box(ax, center, size, label, type='layer', sub_label="") -> patches.FancyBboxPatch:
    x, y = center
    w, h = size
    
    # Color config
    fc = COLORS.get(type, '#FFFFFF')
    ec = COLORS.get(f'border_{type}', '#000000')
    
    # Shadow offset
    shadow = patches.FancyBboxPatch(
        (x - w/2 + 0.02, y - h/2 - 0.02), w, h,
        boxstyle=f"round,pad=0.0,rounding_size=0.1",
        facecolor='#000000', alpha=0.1, zorder=1
    )
    ax.add_patch(shadow)

    # Main box
    box = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.0,rounding_size=0.1",
        facecolor=fc, edgecolor=ec, linewidth=2, zorder=2
    )
    ax.add_patch(box)
    
    # Text
    ax.text(x, y + 0.05 if sub_label else y, label, ha='center', va='center', fontsize=10, weight='bold', zorder=3)
    if sub_label:
        ax.text(x, y - 0.15, sub_label, ha='center', va='center', fontsize=8, color='#444', zorder=3)
        
    return box

def draw_arrow(ax, start, end) -> None:
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color="#444", lw=2, shrinkA=5, shrinkB=5),
                zorder=1)

def viz_curve_mlp() -> None:
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("CurveMLP Architecture", pad=20)
    
    nodes = [
        ("Input Features", "vector", 'input'),
        ("Linear 128", "BN + ReLU", 'layer'),
        ("Linear 256", "BN + ReLU", 'layer'),
        ("Linear 128", "BN + ReLU", 'layer'),
        ("Control Points", "Linear Head", 'special'),
        ("Monotonicity", "Constraint", 'special'),
        ("Interpolation", "Linear", 'special'),
        ("Output LUT", "Density Curve", 'output')
    ]
    
    y_start = 9
    spacing = 1.2
    
    positions = []
    
    for i, (label, sub, type_) in enumerate(nodes):
        pos = (3, y_start - i * spacing)
        positions.append(pos)
        draw_box(ax, pos, (3, 0.8), label, type=type_, sub_label=sub)
        
        if i > 0:
            draw_arrow(ax, positions[i-1], (pos[0], pos[1] + 0.4))
            
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arch_curve_mlp.png", dpi=150, bbox_inches='tight')
    plt.close()

def viz_curve_cnn() -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("CurveCNN Architecture", pad=20)
    
    nodes = [
        ("Input", "vector", 'input'),
        ("Proj", "Linear", 'layer'),
        ("Conv1D", "k=3, c=64", 'conv'),
        ("Conv1D", "k=3, c=128", 'conv'),
        ("Conv1D", "k=3, c=64", 'conv'),
        ("OutProj", "1x1 Conv", 'layer'),
        ("Mono", "Constraint", 'special'),
        ("Output", "LUT", 'output')
    ]
    
    x_start = 1
    spacing = 2.5
    
    positions = []
    for i, (label, sub, type_) in enumerate(nodes):
        pos = (x_start + i * spacing, 2)
        positions.append(pos)
        draw_box(ax, pos, (2, 1), label, type=type_, sub_label=sub)
        
        if i > 0:
            draw_arrow(ax, (positions[i-1][0]+1, 2), (pos[0]-1, 2))
            
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arch_curve_cnn.png", dpi=150, bbox_inches='tight')
    plt.close()

def viz_content_aware_net() -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax.set_aspect('equal') # Disable equal aspect for U-Net better scaling
    ax.axis('off')
    ax.set_title("ContentAwareCurveNet (U-Net)", pad=20)
    
    # Coordinates
    levels = 4
    x_enc = [2, 3, 4, 5]
    x_dec = [8, 7, 6, 5] # Aligned with encoders
    y_levels = [8, 6, 4, 2] # Depth
    
    # Input
    draw_box(ax, (2, 9.5), (2, 0.8), "Input Image", 'input')
    draw_arrow(ax, (2, 9.1), (2, 8.4))
    
    enc_pos = []
    dec_pos = []
    
    # Encoder
    for i in range(levels):
        pos = (2 + i*0.8, y_levels[i])
        enc_pos.append(pos)
        label = f"Enc {i+1}"
        sub = f"{32*(2**i)} ch"
        draw_box(ax, pos, (1.5, 1), label, 'layer', sub)
        
        if i < levels - 1:
            # Pooling arrow down-right
            next_pos = (2 + (i+1)*0.8, y_levels[i+1] + 0.5)
            draw_arrow(ax, (pos[0], pos[1]-0.5), next_pos)

    # Bottleneck
    bot_pos = (2 + (levels)*0.8, 0)
    # Actually let's make it simpler layout: V shape
    
    # Re-doing layout for V-shape U-Net
    
    # Encoder stream
    e_pos = [(2, 8), (2, 6), (2, 4), (2, 2)]
    for i, pos in enumerate(e_pos):
        draw_box(ax, pos, (2, 1.2), f"Encoder {i+1}", 'layer', f"{32*(2**i)} ch")
        if i < 3:
            draw_arrow(ax, (pos[0], pos[1]-0.6), (e_pos[i+1][0], e_pos[i+1][1]+0.6))
            ax.text(pos[0]-0.2, pos[1]-1, "Pool", fontsize=8, color='#666')

    # Bottleneck
    b_pos = (5, 1)
    draw_box(ax, b_pos, (2, 1.2), "Bottleneck", 'layer', "512 ch")
    draw_arrow(ax, (e_pos[-1][0], e_pos[-1][1]-0.6), (b_pos[0]-1, b_pos[1]))
    
    # Decoder stream
    d_pos = [(8, 2), (8, 4), (8, 6), (8, 8)]
    for i, pos in enumerate(d_pos):
        ch = 256 // (2**i)
        draw_box(ax, pos, (2, 1.2), f"Decoder {i+1}", 'layer', f"{ch} ch")
        if i == 0:
             draw_arrow(ax, (b_pos[0]+1, b_pos[1]), (pos[0], pos[1]-0.6))
        else:
             draw_arrow(ax, (d_pos[i-1][0], d_pos[i-1][1]+0.6), (pos[0], pos[1]-0.6))
             
        # Skip connections
        skip_start = e_pos[3-i]
        draw_arrow(ax, (skip_start[0]+1, skip_start[1]), (pos[0]-1, pos[1]))
        ax.text((skip_start[0]+pos[0])/2, pos[1]+0.1, "Skip", fontsize=8, color='#999', ha='center')

    # Output
    out_pos = (8, 9.5)
    draw_box(ax, out_pos, (2, 0.8), "Adjustment Map", 'output')
    draw_arrow(ax, (d_pos[-1][0], d_pos[-1][1]+0.6), (out_pos[0], out_pos[1]-0.4))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arch_unet.png", dpi=150, bbox_inches='tight')
    plt.close()

def viz_uniformity_net() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("UniformityCorrectionNet", pad=20)
    
    nodes = [
        ("Input", "", 'input'),
        ("Conv1", "3x3, 16ch", 'layer'),
        ("Conv2", "3x3, 16ch", 'layer'),
        ("Conv3", "3x3, 1ch", 'layer'),
        ("Smoothing", "Gaussian\nKernel", 'special'),
        ("Output", "Map", 'output')
    ]
    
    x_start = 1.5
    spacing = 2.5
    
    positions = []
    for i, (label, sub, type_) in enumerate(nodes):
        pos = (x_start + i * spacing, 2)
        positions.append(pos)
        draw_box(ax, pos, (2, 1.2), label, type=type_, sub_label=sub)
        
        if i > 0:
            draw_arrow(ax, (positions[i-1][0]+1, 2), (pos[0]-1, 2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arch_uniformity.png", dpi=150, bbox_inches='tight')
    plt.close()

def main() -> None:
    print("Generating visualizations...")
    viz_curve_mlp()
    viz_curve_cnn()
    viz_content_aware_net()
    viz_uniformity_net()
    print(f"Saved images to {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
