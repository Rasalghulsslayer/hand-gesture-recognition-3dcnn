import numpy as np
import torch
from pathlib import Path
import cv2
from tqdm import tqdm

# Paramètres
TARGET_FRAMES = 32
H_HRN, W_HRN = 57, 125
H_LRN, W_LRN = 28, 62

def temporal_resampling(volume):
    if volume.shape[0] == TARGET_FRAMES: return volume
    indices = np.linspace(0, volume.shape[0] - 1, TARGET_FRAMES)
    indices = np.round(indices).astype(int)
    return volume[indices]

def normalize(volume):
    mean, std = np.mean(volume), np.std(volume)
    return (volume - mean) / std if std > 0 else volume - mean

def rebuild_dataset():
    raw_path = Path("data/raw/video_array_raw") # Vérifie ce chemin !
    if not raw_path.exists():
        raw_path = Path("data/raw") # Fallback
    
    out_path = Path("data/processed")
    out_path.mkdir(exist_ok=True, parents=True)
    
    # 1. Charger les labels
    labels_file = Path("data/raw/labels.npy")
    if not labels_file.exists():
        print("❌ CRITIQUE: labels.npy introuvable dans data/raw/")
        return
        
    print("Chargement et conversion des labels...")
    one_hot = np.load(labels_file)
    labels_indices = np.argmax(one_hot, axis=1)
    
    # Sauvegarde des labels PROPRES
    torch.save(torch.tensor(labels_indices), out_path / "labels.pt")
    
    print(f"Labels prêts. Total: {len(labels_indices)}")
    
    # 2. Traitement des vidéos en suivant STRICTEMENT l'index
    # On boucle sur les indices des labels (0 à N)
    # Et on cherche le fichier correspondant "0.npy", "1.npy"...
    
    count_ok = 0
    count_missing = 0
    
    print("Reconstruction du dataset aligné...")
    for idx in tqdm(range(len(labels_indices))):
        
        # On cherche le fichier qui porte EXACTEMENT ce numéro
        file_name = f"{idx}.npy"
        file_path = raw_path / file_name
        
        if not file_path.exists():
            count_missing += 1
            continue
            
        # Chargement
        try:
            video_data = np.load(file_path)
            
            # Resample Temporel
            video_data = temporal_resampling(video_data)
            
            frames_hrn = []
            frames_lrn = []
            
            # Resize Spatial
            for i in range(TARGET_FRAMES):
                frm = video_data[i]
                frames_hrn.append(cv2.resize(frm, (W_HRN, H_HRN)))
                frames_lrn.append(cv2.resize(frm, (W_LRN, H_LRN)))
                
            # Normalisation & Tensor HRN
            vol_hrn = normalize(np.array(frames_hrn))
            ten_hrn = torch.FloatTensor(vol_hrn).unsqueeze(0)
            torch.save(ten_hrn, out_path / f"hrn_{idx}.pt")
            
            # Normalisation & Tensor LRN
            vol_lrn = normalize(np.array(frames_lrn))
            ten_lrn = torch.FloatTensor(vol_lrn).unsqueeze(0)
            torch.save(ten_lrn, out_path / f"lrn_{idx}.pt")
            
            count_ok += 1
            
        except Exception as e:
            print(f"Erreur sur {file_name}: {e}")

    print(f"\n✅ Terminé !")
    print(f"Fichiers traités : {count_ok}")
    print(f"Fichiers manquants : {count_missing}")
    print(f"Labels totaux : {len(labels_indices)}")
    print("Les indices sont maintenant garantis alignés (hrn_X.pt correspond à label[X]).")

if __name__ == "__main__":
    rebuild_dataset()