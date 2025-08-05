#!/usr/bin/env python3
"""
Mosaic-Style Batch Processing Architecture Visualization
- Process multiple records as batch to create one mosaic image
- Each record becomes one row in the mosaic
- Server reconstructs entire batch from single mosaic image
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")

def visualize_mosaic_architecture():
    """Mosaic-style batch processing architecture visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Mosaic-Style Batch Processing Architecture', fontsize=20, fontweight='bold', y=0.95)
    
    # ===== 1. Multiple Records (Batch) =====
    ax1 = axes[0, 0]
    ax1.set_title('Step 1: Batch of Records (Multiple People)', fontsize=14, fontweight='bold')
    
    # Show multiple records
    colors = plt.cm.Set3(np.linspace(0, 1, 19))
    
    # Sample batch of 4 people
    batch_data = [
        ['Person 1:', 'voter_id: 5168123', 'first_name: joseph', '...age: 53'],
        ['Person 2:', 'voter_id: 7284951', 'first_name: sarah', '...age: 34'], 
        ['Person 3:', 'voter_id: 3847562', 'first_name: mike', '...age: 67'],
        ['Person 4:', 'voter_id: 9105738', 'first_name: lisa', '...age: 29']
    ]
    
    y_start = 0.85
    for person_idx, person_data in enumerate(batch_data):
        # Person header
        rect = FancyBboxPatch((0.05, y_start - person_idx*0.2), 0.9, 0.04, 
                             boxstyle="round,pad=0.01",
                             facecolor=f'C{person_idx}', 
                             edgecolor='black',
                             alpha=0.9)
        ax1.add_patch(rect)
        ax1.text(0.1, y_start - person_idx*0.2 + 0.02, person_data[0], 
                fontsize=11, fontweight='bold')
        
        # Person features (show few sample features)
        for feat_idx, feat in enumerate(person_data[1:]):
            rect = FancyBboxPatch((0.1, y_start - person_idx*0.2 - (feat_idx+1)*0.03), 0.8, 0.025, 
                                 boxstyle="round,pad=0.005",
                                 facecolor=colors[feat_idx], 
                                 edgecolor='black',
                                 alpha=0.6)
            ax1.add_patch(rect)
            ax1.text(0.12, y_start - person_idx*0.2 - (feat_idx+1)*0.03 + 0.01, feat, 
                    fontsize=8, fontweight='bold')
    
    ax1.text(0.5, 0.02, 'Batch of records → Process together as one unit', 
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # ===== 2. Batch Matrix Formation =====
    ax2 = axes[0, 1]
    ax2.set_title('Step 2: Batch → Matrix (Each Person = One Row)', fontsize=14, fontweight='bold')
    
    # Show batch matrix formation
    ax2.text(0.5, 0.92, '(batch_size=4, features=19)', 
             ha='center', fontsize=11, fontweight='bold')
    
    # Create 4x19 matrix visualization
    matrix_x = 0.1
    matrix_y = 0.3
    cell_width = 0.035
    cell_height = 0.08
    
    # Labels for people
    person_labels = ['Person 1', 'Person 2', 'Person 3', 'Person 4']
    feature_labels = ['v_id', 'name', 'age', 'gender', '...', 'date']
    
    for i in range(4):  # 4 people
        for j in range(19):  # 19 features
            # Color based on person - ensure alpha stays in [0,1] range
            intensity = 0.3 + (i * 0.1) + np.random.rand() * 0.2
            intensity = min(intensity, 1.0)  # Cap at 1.0
            rect = patches.Rectangle((matrix_x + j*cell_width, matrix_y + i*cell_height), 
                                   cell_width, cell_height,
                                   facecolor=f'C{i}',
                                   edgecolor='white',
                                   linewidth=1,
                                   alpha=intensity)
            ax2.add_patch(rect)
            
            # Add some sample values
            if j < 6:  # Show first 6 features
                if j < len(feature_labels):
                    ax2.text(matrix_x + j*cell_width + cell_width/2, 
                            matrix_y + i*cell_height + cell_height/2, 
                            f'{np.random.rand():.1f}', 
                            ha='center', va='center', fontsize=6, fontweight='bold')
    
    # Row labels (people)
    for i, label in enumerate(person_labels):
        ax2.text(matrix_x - 0.02, matrix_y + i*cell_height + cell_height/2, 
                label, ha='right', va='center', fontsize=9, fontweight='bold', color=f'C{i}')
    
    # Column labels (features) - show a few
    for j, label in enumerate(feature_labels):
        if j < 6:
            ax2.text(matrix_x + j*cell_width + cell_width/2, matrix_y - 0.02, 
                    label, ha='center', va='top', fontsize=8, rotation=45)
    
    ax2.text(0.5, 0.15, 'Each row = One person (19 features)\nEach column = One feature across all people', 
             ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax2.annotate('', xy=(0.5, 0.25), xytext=(0.5, 0.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax2.text(0.52, 0.55, 'Stack into\nMatrix', ha='left', va='center', 
             fontsize=10, fontweight='bold', color='red')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # ===== 3. Matrix → Single Mosaic Image =====
    ax3 = axes[1, 0]
    ax3.set_title('Step 3: Matrix → Single Mosaic Image', fontsize=14, fontweight='bold')
    
    # Input matrix representation
    input_rect = patches.Rectangle((0.05, 0.7), 0.25, 0.2, 
                                 facecolor='lightblue', 
                                 edgecolor='black',
                                 linewidth=2)
    ax3.add_patch(input_rect)
    ax3.text(0.175, 0.8, 'Matrix\n(4×19)', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    
    # Neural network processing
    nn_positions = [(0.4, 0.75), (0.55, 0.7)]
    nn_labels = ['Neural\nNetwork', 'Conv2D\nLayers']
    
    for i, (pos, label) in enumerate(zip(nn_positions, nn_labels)):
        rect = patches.Rectangle(pos, 0.12, 0.15, 
                               facecolor=f'C{i+1}', 
                               edgecolor='black',
                               alpha=0.7)
        ax3.add_patch(rect)
        ax3.text(pos[0] + 0.06, pos[1] + 0.075, label, 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrows
        if i == 0:
            ax3.annotate('', xy=(pos[0], pos[1] + 0.075), 
                        xytext=(0.3, 0.8),
                        arrowprops=dict(arrowstyle='->', lw=2))
        else:
            prev_pos = nn_positions[i-1]
            ax3.annotate('', xy=(pos[0], pos[1] + 0.075), 
                        xytext=(prev_pos[0] + 0.12, prev_pos[1] + 0.075),
                        arrowprops=dict(arrowstyle='->', lw=2))
    
    # Single mosaic image output
    mosaic_rect = patches.Rectangle((0.05, 0.2), 0.4, 0.4, 
                                  facecolor='lightgreen', 
                                  edgecolor='black',
                                  linewidth=2)
    ax3.add_patch(mosaic_rect)
    
    # Create mosaic pattern showing different people's data
    for i in range(20):
        for j in range(20):
            x = 0.05 + i * 0.02
            y = 0.2 + j * 0.02
            # Different patterns for different "rows" (people)
            person_row = j // 5  # Which person this pixel represents
            intensity = np.sin(i*0.3 + person_row) * np.cos(j*0.2)
            color_val = (intensity + 1) / 2
            rect = patches.Rectangle((x, y), 0.02, 0.02,
                                   facecolor=plt.cm.Set1(person_row % 4),
                                   alpha=color_val * 0.8 + 0.2,
                                   edgecolor='none')
            ax3.add_patch(rect)
    
    ax3.text(0.25, 0.15, 'Single Mosaic Image\n(Contains ALL people)', 
             ha='center', fontsize=11, fontweight='bold')
    
    # Arrow to server
    ax3.annotate('', xy=(0.8, 0.4), xytext=(0.45, 0.4),
                arrowprops=dict(arrowstyle='->', lw=4, color='purple'))
    ax3.text(0.625, 0.45, 'Send to Server', ha='center', fontsize=11, 
             fontweight='bold', color='purple')
    
    # Final arrow from matrix to mosaic
    ax3.annotate('', xy=(0.25, 0.6), xytext=(0.67, 0.75),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # ===== 4. Server: Mosaic → Batch Reconstruction =====
    ax4 = axes[1, 1]
    ax4.set_title('Step 4: Server Reconstructs ALL People from Mosaic', fontsize=14, fontweight='bold')
    
    # Received mosaic image
    input_img_rect = patches.Rectangle((0.05, 0.7), 0.25, 0.25, 
                                     facecolor='lightgreen', 
                                     edgecolor='black',
                                     linewidth=2)
    ax4.add_patch(input_img_rect)
    ax4.text(0.175, 0.825, 'Received\nMosaic', ha='center', va='center', 
             fontsize=10, fontweight='bold')
    
    # Decoder layers
    decoder_positions = [(0.4, 0.8), (0.55, 0.75), (0.7, 0.7)]
    decoder_labels = ['CNN\nDecoder', 'Batch\nSplitter', 'Output\nLayers']
    
    for i, (pos, label) in enumerate(zip(decoder_positions, decoder_labels)):
        rect = patches.Rectangle(pos, 0.12, 0.15, 
                               facecolor=f'C{i+4}', 
                               edgecolor='black',
                               alpha=0.7)
        ax4.add_patch(rect)
        ax4.text(pos[0] + 0.06, pos[1] + 0.075, label, 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrows
        if i == 0:
            ax4.annotate('', xy=(pos[0], pos[1] + 0.075), 
                        xytext=(0.3, 0.825),
                        arrowprops=dict(arrowstyle='->', lw=2))
        else:
            prev_pos = decoder_positions[i-1]
            ax4.annotate('', xy=(pos[0], pos[1] + 0.075), 
                        xytext=(prev_pos[0] + 0.12, prev_pos[1] + 0.075),
                        arrowprops=dict(arrowstyle='->', lw=2))
    
    # Reconstructed batch matrix
    restored_y = 0.15
    matrix_small_x = 0.1
    cell_small_width = 0.03
    cell_small_height = 0.06
    
    for i in range(4):  # 4 people
        for j in range(19):  # 19 features  
            if j < 10:  # Show first 10 features for space
                rect = patches.Rectangle((matrix_small_x + j*cell_small_width, restored_y + i*cell_small_height), 
                                       cell_small_width, cell_small_height,
                                       facecolor=f'C{i}', 
                                       edgecolor='white',
                                       linewidth=0.5,
                                       alpha=0.8)
                ax4.add_patch(rect)
    
    # Person labels for reconstructed data
    for i in range(4):
        ax4.text(matrix_small_x - 0.02, restored_y + i*cell_small_height + cell_small_height/2, 
                f'P{i+1}', ha='right', va='center', fontsize=8, fontweight='bold', color=f'C{i}')
    
    # Final arrow
    ax4.annotate('', xy=(0.5, 0.4), xytext=(0.76, 0.7),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    
    ax4.text(0.5, 0.1, 'Reconstructed: ALL 4 people × 19 features', ha='center', fontsize=11, fontweight='bold')
    ax4.text(0.5, 0.05, 'One mosaic → Multiple records!', ha='center', fontsize=10, style='italic', color='red')
    
    # Accuracy display
    accuracy_rect = patches.Rectangle((0.6, 0.15), 0.35, 0.25, 
                                    facecolor='yellow', 
                                    edgecolor='red',
                                    linewidth=2)
    ax4.add_patch(accuracy_rect)
    ax4.text(0.775, 0.275, 'More Information!\n\n• Server gets richer data\n• Better learning context\n• Higher accuracy', 
             ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('mosaic_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Mosaic-style batch processing architecture visualization complete!")
    print("   - Process multiple people as batch to create one mosaic image")
    print("   - Each person becomes one row in the mosaic")
    print("   - Server gets richer information for better learning")

def create_detailed_mosaic_visualization():
    """More detailed mosaic structure visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Detailed Mosaic Structure: Batch → Single Image → Batch', fontsize=16, fontweight='bold')
    
    # ===== Left: Batch Matrix =====
    ax1.set_title('Input: Batch Matrix (4 people × 19 features)', fontsize=14, fontweight='bold')
    
    # Create detailed 4x19 matrix
    batch_data = np.random.rand(4, 19)
    
    # Display matrix with colormap
    im1 = ax1.imshow(batch_data, cmap='viridis', aspect='auto')
    
    # Set axis labels
    feature_names = [
        'voter_id', 'voter_reg_num', 'name_prefix', 'first_name', 'middle_name',
        'last_name', 'name_suffix', 'age', 'gender', 'race', 'ethnic',
        'street_address', 'city', 'state', 'zip_code', 'full_phone_num',
        'birth_place', 'register_date', 'download_month'
    ]
    
    ax1.set_xticks(range(19))
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(['Person 1', 'Person 2', 'Person 3', 'Person 4'])
    
    # Show grid
    ax1.set_xticks(np.arange(-0.5, 19, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax1.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    # Add values as text (show some)
    for i in range(4):
        for j in range(0, 19, 3):  # Show every 3rd value to avoid clutter
            ax1.text(j, i, f'{batch_data[i,j]:.2f}', ha='center', va='center', 
                    fontweight='bold', color='white' if batch_data[i,j] > 0.5 else 'black', fontsize=8)
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Normalized Values [0-1]', rotation=270, labelpad=20)
    
    # ===== Right: Single Mosaic Image =====
    ax2.set_title('Output: Single Mosaic Image (64×64×3)', fontsize=14, fontweight='bold')
    
    # Create mosaic image simulation
    mosaic_image = np.zeros((64, 64, 3))
    
    # Fill different regions with different patterns representing different people
    for person in range(4):
        # Each person gets a section of the image
        start_row = person * 16
        end_row = (person + 1) * 16
        
        for i in range(start_row, end_row):
            for j in range(64):
                # Create pattern based on person and position
                r = np.sin(i * 0.1 + person) * 0.5 + 0.5
                g = np.cos(j * 0.1 + person) * 0.5 + 0.5  
                b = np.sin((i+j) * 0.05 + person) * 0.5 + 0.5
                
                mosaic_image[i, j] = [r, g, b]
    
    ax2.imshow(mosaic_image)
    
    # Add person region labels
    for person in range(4):
        start_row = person * 16
        ax2.text(-2, start_row + 8, f'Person {person+1}', rotation=90, 
                ha='center', va='center', fontweight='bold', color=f'C{person}', fontsize=12)
        # Add horizontal line to separate regions
        if person < 3:
            ax2.axhline(y=(person+1)*16 - 0.5, color='white', linewidth=2)
    
    ax2.set_xlabel('Width (64 pixels)')
    ax2.set_ylabel('Height (64 pixels)')
    
    # Add explanation
    ax2.text(32, -8, 'Each person occupies 16 rows\nAll information compressed into single image', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('detailed_mosaic_structure.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Detailed mosaic structure visualization complete!")

if __name__ == "__main__":
    print("🎨 Mosaic-style batch processing architecture visualization starting...")
    visualize_mosaic_architecture()
    print("\n" + "="*60)
    create_detailed_mosaic_visualization()
