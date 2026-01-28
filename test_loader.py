import torch
from torch.utils.data import DataLoader
from training.dataset import VivaDataset
from training.augmentations import Compose, SpatialAugmentation, TemporalElasticDeformation, ToTensor

def test_pipeline():
    # Définition des transformations (Pipeline complet du papier)
    transforms = Compose([
        SpatialAugmentation(prob=0.5),           # Rotation, Scale, etc.
        TemporalElasticDeformation(prob=0.5),    # La fameuse TED
        ToTensor()                               # Retourne un Tensor propre
    ])
    
    # Création du Dataset (On prend les 10 premiers indices pour tester)
    # Assure-toi que data/processed contient bien des fichiers
    dataset = VivaDataset(
        root_dir="data/processed", 
        mode='hrn', 
        indices=range(10), 
        transform=transforms
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("Test du chargement d'un batch avec augmentations...")
    for videos, labels in loader:
        print(f"Batch Shape: {videos.shape}")
        print(f"Labels: {labels}")
        
        # Vérification critique : Dimensions
        # HRN attend (Batch, 1, 32, 57, 125)
        assert videos.shape[1] == 1, "Erreur Canal (doit être 1)"
        assert videos.shape[2] == 32, "Erreur Profondeur (doit être 32 frames)"
        assert videos.shape[3] == 57, "Erreur Hauteur"
        assert videos.shape[4] == 125, "Erreur Largeur"
        
        print("✅ Batch chargé et dimensions valides !")
        break

if __name__ == "__main__":
    test_pipeline()