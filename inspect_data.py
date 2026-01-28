import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def inspect_data():
    data_dir = Path("data/processed")
    
    # 1. Vérifications de base
    if not data_dir.exists():
        print("❌ Erreur : Dossier data/processed introuvable.")
        return

    labels_file = data_dir / "labels.pt"
    if not labels_file.exists():
        print("❌ Erreur : labels.pt introuvable.")
        return
    
    all_labels = torch.load(labels_file)
    print(f"📊 Total Labels : {len(all_labels)}")
    print(f"   Classes (min-max) : {all_labels.min().item()} - {all_labels.max().item()}")
    
    # 2. Trouver les fichiers HRN
    files = list(data_dir.glob("hrn_*.pt"))
    print(f"📂 Fichiers HRN trouvés : {len(files)}")
    
    if not files: return

    # 3. Analyser 8 échantillons aléatoires
    sample_size = min(8, len(files))
    samples = random.sample(files, sample_size)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Ce que voit le réseau (HRN - Frame 16)", fontsize=16)
    
    print("\n--- Statistiques des Tenseurs ---")
    for i, fpath in enumerate(samples):
        # Chargement
        tensor = torch.load(fpath) # Shape (1, 32, 57, 125)
        
        # Récupération du label
        idx = int(fpath.stem.split('_')[1])
        label = all_labels[idx].item()
        
        data = tensor.numpy()
        d_min, d_max, d_mean, d_std = data.min(), data.max(), data.mean(), data.std()
        
        print(f"Fichier: {fpath.name:<15} | Label: {label:<2} | Min: {d_min:>6.2f} | Max: {d_max:>6.2f} | Mean: {d_mean:>5.2f} | Std: {d_std:>4.2f}")
        
        # Visualisation de la frame du milieu
        mid_frame = data[0, 16, :, :] 
        
        # Normalisation visuelle [0, 1] pour matplotlib
        if d_max > d_min:
            img_show = (mid_frame - d_min) / (d_max - d_min)
        else:
            img_show = mid_frame
            
        ax = axes[i//4, i%4]
        ax.imshow(img_show, cmap='gray')
        ax.set_title(f"Label: {label}\nVal: [{d_min:.1f}, {d_max:.1f}]")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("data_inspection.png")
    print("\n✅ Image générée : 'data_inspection.png'")
    print("👉 Ouvre cette image pour vérifier si on distingue bien une main.")

if __name__ == "__main__":
    inspect_data()