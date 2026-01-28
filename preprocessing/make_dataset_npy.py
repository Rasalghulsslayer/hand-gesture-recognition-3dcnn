import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Paramètres fixes du papier
TARGET_FRAMES = 32
TARGET_HEIGHT_HRN = 57
TARGET_WIDTH_HRN = 125
TARGET_HEIGHT_LRN = 28
TARGET_WIDTH_LRN = 62

def temporal_resampling(volume, target_frames=TARGET_FRAMES):
    """
    Transforme une vidéo de N frames en 32 frames via Nearest Neighbor Interpolation.
    Gère le cas où N > 32 (drop frames) et N < 32 (repeat frames).
    """
    num_frames = volume.shape[0]
    if num_frames == target_frames:
        return volume
        
    # On crée des indices espacés uniformément (ex: 0, 1.2, 2.4...)
    indices = np.linspace(0, num_frames - 1, target_frames)
    indices = np.round(indices).astype(int)
    
    return volume[indices] # Sélectionne les frames correspondantes

def normalize_volume(volume):
    """Normalisation : Moyenne zéro, Variance 1"""
    mean = np.mean(volume)
    std = np.std(volume)
    if std > 0:
        return (volume - mean) / std
    return volume - mean

def process_npy_data(source_dir, output_dir):
    """
    Docstring pour process_npy_data
    
    :param source_dir: Description
    :param output_dir: Description
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Gestion des Labels (Identique)
    labels_file = source_path / "labels.npy"
    if labels_file.exists():
        one_hot_labels = np.load(labels_file)
        labels_indices = np.argmax(one_hot_labels, axis=1)
        torch.save(torch.tensor(labels_indices), output_path / "labels.pt")
        print(f"Labels traités : {len(labels_indices)} échantillons.")
    else:
        print("ATTENTION: labels.npy introuvable. Seuls les features seront générés.")

    # 2. Traitement des vidéos depuis 'video_array_raw'
    # On cherche dans le dossier que tu as mentionné
    video_dir = source_path / "video_array_raw" 
    if not video_dir.exists():
        # Fallback si l'utilisateur a mis les fichiers directement dans source
        video_dir = source_path
        
    files = list(video_dir.glob("*.npy"))
    print(f"Traitement de {len(files)} fichiers depuis {video_dir}...")
    
    for file_path in tqdm(files):
        if file_path.name == "labels.npy": continue # Skip labels file
        
        try:
            # Le nom du fichier est l'index (ex: "2.npy")
            idx = int(file_path.stem)
        except ValueError:
            continue 

        # Chargement : Shape (N_frames, 115, 250)
        video_data = np.load(file_path)
        
        # --- ÉTAPE CRITIQUE : NNI ---
        # On force la vidéo à faire 32 frames, quelle que soit sa longueur initiale
        video_data = temporal_resampling(video_data, target_frames=TARGET_FRAMES)
        
        processed_frames_hrn = []
        processed_frames_lrn = [] # On prépare aussi le LRN
        
        for i in range(TARGET_FRAMES):
            frame = video_data[i]
            
            # Spatial Resize
            frame_hrn = cv2.resize(frame, (TARGET_WIDTH_HRN, TARGET_HEIGHT_HRN))
            frame_lrn = cv2.resize(frame, (TARGET_WIDTH_LRN, TARGET_HEIGHT_LRN))
            
            processed_frames_hrn.append(frame_hrn)
            processed_frames_lrn.append(frame_lrn)
            
        # Normalisation & Conversion Tensor
        # HRN
        vol_hrn = np.array(processed_frames_hrn)
        vol_hrn = normalize_volume(vol_hrn)
        tensor_hrn = torch.FloatTensor(vol_hrn).unsqueeze(0) # (1, 32, 57, 125)
        
        # LRN
        vol_lrn = np.array(processed_frames_lrn)
        vol_lrn = normalize_volume(vol_lrn)
        tensor_lrn = torch.FloatTensor(vol_lrn).unsqueeze(0) # (1, 32, 28, 62)

        # Sauvegarde (on sauvegarde les deux résolutions dans le même dossier)
        # On utilise un suffixe pour les distinguer
        torch.save(tensor_hrn, output_path / f"hrn_{idx}.pt")
        torch.save(tensor_lrn, output_path / f"lrn_{idx}.pt")

if __name__ == "__main__":
    # Assure-toi que tes données sont bien dans data/raw/video_array_raw/
    process_npy_data("data/raw", "data/processed")