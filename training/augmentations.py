import numpy as np
import cv2
import torch
import random
from scipy.interpolate import interp1d

class Compose:
    """Enchaîne plusieurs transformations."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)
        return video

class ToTensor:
    """Convertit le numpy array en Tensor PyTorch (C, D, H, W)."""
    def __call__(self, video):
        # video est supposé être (D, H, W) ou (D, H, W, C)
        # On veut sortir (C, D, H, W)
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        
        # Si input est (Depth, Height, Width), on ajoute la dimension Channel -> (1, D, H, W)
        if video.dim() == 3:
            video = video.unsqueeze(0)
            
        return video.float()

# --- Augmentations Spatiales (Appliquées uniformément sur toutes les frames) ---

class SpatialAugmentation:
    """
    Applique Rotation, Scale, et Translation.
    Papier : Rotation (+/- 10 deg), Scale (+/- 30%), Translation (+/- 4px X, +/- 8px Y)
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, video):
        """
        video: Tensor ou Numpy (Depth, Height, Width) ou (C, D, H, W)
        Pour simplifier ici on attend du Numpy (D, H, W) normalisé
        """
        if random.random() > self.prob:
            return video

        # On travaille en numpy pour utiliser OpenCV
        if torch.is_tensor(video):
            was_tensor = True
            device = video.device
            video = video.numpy()
        else:
            was_tensor = False
            
        # S'assurer qu'on a (Depth, Height, Width)
        # Si on a (C, D, H, W), on prend le canal 0 pour l'instant (car on a que du gradient)
        if video.ndim == 4:
            vid_data = video[0] 
        else:
            vid_data = video

        depth, h, w = vid_data.shape
        
        # 1. Tirage des paramètres aléatoires (Fixes pour toute la vidéo)
        angle = random.uniform(-10, 10)
        scale = random.uniform(0.7, 1.3)
        tx = random.uniform(-4, 4)
        ty = random.uniform(-8, 8)

        # Matrice de transformation affine
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        new_video = []
        for i in range(depth):
            frame = vid_data[i]
            # warpAffine gère l'interpolation
            augmented_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            new_video.append(augmented_frame)

        new_video = np.array(new_video)
        
        # Reconstruction du format original
        if video.ndim == 4:
            new_video = np.expand_dims(new_video, axis=0)

        if was_tensor:
            return torch.from_numpy(new_video).to(device)
            
        return new_video

# --- Augmentation Temporelle (La partie unique du papier) ---

class TemporalElasticDeformation:
    """
    [cite_start]Temporal Elastic Deformation (TED) [cite: 216-238]
    Déforme le temps selon une courbe polynomiale.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, video):
        if random.random() > self.prob:
            return video

        if torch.is_tensor(video):
            was_tensor = True
            data = video.numpy()
        else:
            was_tensor = False
            data = video
            
        # On extrait la dimension temporelle D
        # Format attendu (C, D, H, W) ou (D, H, W)
        if data.ndim == 4:
            D = data.shape[1]
        else:
            D = data.shape[0]

        # Paramètres de la déformation (Eq. 6 du papier)
        M = D / 2.0 # Milieu
        
        # n ~ Normal(M, 4)
        n = np.random.normal(M, 4)
        n = np.clip(n, 0, D-1) # Borné
        
        # m ~ Normal(n, 4 * (1 - |n-M|/M))
        sigma_m = 4 * (1 - abs(n - M) / M)
        if sigma_m <= 0.1: sigma_m = 0.1 # Sécurité
        m = np.random.normal(n, sigma_m)
        m = np.clip(m, 0, D-1)

        # Interpolation polynomiale d'ordre 2 (Quadratique)
        # On veut mapper les indices d'entrée [0, n, D-1] vers [0, m, D-1]
        src_points = [0, n, D-1]
        dst_points = [0, m, D-1]
        
        # Fit polynomial (indices -> temps déformé)
        # On cherche P(t) tel que P(0)=0, P(n)=m, P(D-1)=D-1
        try:
            poly = np.polyfit(src_points, dst_points, 2)
            func = np.poly1d(poly)
            
            # Générer les nouveaux indices d'échantillonnage
            original_indices = np.arange(D)
            new_indices = func(original_indices)
            
            # Clip pour rester dans les bornes
            new_indices = np.clip(new_indices, 0, D-1)
            
            # Ré-échantillonnage (Interpolation linéaire sur l'axe temporel)
            # Pour chaque pixel (h,w), on doit interpoler la valeur aux temps new_indices
            # C'est lourd pixel par pixel. On fait frame par frame avec pondération.
            
            # Méthode rapide : Nearest Neighbor sur les indices (comme le papier NNI)
            # [cite_start]Le papier dit "interpolated using ... nearest-neighbor interpolation" [cite: 238]
            indices_integers = np.round(new_indices).astype(int)
            
            if data.ndim == 4:
                new_data = data[:, indices_integers, :, :]
            else:
                new_data = data[indices_integers, :, :]
                
            if was_tensor:
                return torch.from_numpy(new_data)
            return new_data

        except:
            # En cas d'erreur de fit (points colinéaires par ex), on ne fait rien
            if was_tensor: return torch.from_numpy(data)
            return data