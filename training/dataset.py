import torch
from torch.utils.data import Dataset
from pathlib import Path
import os

class VivaDataset(Dataset):
    def __init__(self, root_dir, mode='hrn', indices=None, transform=None):
        """
        Args:
            root_dir (str): Chemin vers data/processed
            mode (str): 'hrn' ou 'lrn'
            indices (list): Liste d'indices forcés (pour le split train/val)
            transform (callable, optional): Augmentations
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform
        
        # 1. Chargement de tous les labels
        self.all_labels = torch.load(self.root_dir / "labels.pt")
        
        # 2. FILTRAGE : On ne garde que les indices dont le fichier existe vraiment sur le disque
        # C'est ici qu'on résout le bug "FileNotFound"
        self.valid_indices = []
        
        # On vérifie l'existence des fichiers une seule fois au démarrage
        # Cela prend une fraction de seconde et évite les crashs
        print(f"🔍 Vérification de l'intégrité du dataset ({mode})...")
        for idx in range(len(self.all_labels)):
            filename = f"{self.mode}_{idx}.pt"
            if (self.root_dir / filename).exists():
                self.valid_indices.append(idx)
        
        print(f"✅ {len(self.valid_indices)} fichiers valides trouvés (sur {len(self.all_labels)} labels).")

        # 3. Gestion du sous-ensemble (Train/Val)
        if indices is None:
            # Si on veut tout le dataset, on prend tous les fichiers valides
            self.indices = self.valid_indices
        else:
            # Si on nous donne une liste précise (via random_split), 
            # on doit s'assurer que ces indices sont bien dans nos fichiers valides
            # On mappe les indices demandés (0..N) vers les indices réels valides
            self.indices = [self.valid_indices[i] for i in indices if i < len(self.valid_indices)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Récupération de l'ID réel du fichier
        real_idx = self.indices[idx]
        
        filename = f"{self.mode}_{real_idx}.pt"
        file_path = self.root_dir / filename
        
        # Chargement
        tensor = torch.load(file_path)
        
        # Augmentations
        if self.transform:
            tensor = self.transform(tensor)
            
        label = self.all_labels[real_idx]
        
        return tensor, label