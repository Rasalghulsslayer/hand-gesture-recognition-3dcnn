import argparse
import torch
from torch.utils.data import DataLoader, random_split
from training.dataset import VivaDataset
from training.augmentations import Compose, SpatialAugmentation, TemporalElasticDeformation, ToTensor
from training.trainer import Trainer
from models.hrn import HRN
from models.lrn import LRN

# Configuration
DATA_DIR = "data/processed"
NUM_CLASSES = 34 # Adapté à ton dataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # C'est ici que la magie opère pour les Mac M1/M2/M3
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

def main(args):
    print(f"--- Entraînement du réseau : {args.network.upper()} ---")
    print(f"Device: {DEVICE}")

    # 1. Préparation des Augmentations (Papier Section 2.5)
    train_transforms = Compose([
        SpatialAugmentation(prob=0.5),        # [cite: 203]
        TemporalElasticDeformation(prob=0.5), # [cite: 216]
        ToTensor()
    ])
    val_transforms = Compose([
        ToTensor() # Pas d'augmentation en validation [cite: 171]
    ])

    # 2. Chargement du Dataset
    full_dataset = VivaDataset(
        root_dir=DATA_DIR, 
        mode=args.network.lower() # 'hrn' ou 'lrn'
    )

    # Split Train/Val (80% / 20% simple pour commencer)
    # Le papier fait du "Leave-One-Subject-Out", plus complexe.
    # Pour un premier test, un split aléatoire est suffisant.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Injection des transforms après le split
    # (Astuce: on hacke l'attribut dataset du subset pour appliquer les transforms)
    # Note: Dans une implémentation stricte, on ferait deux datasets séparés.
    # Ici, pour faire simple, on applique les transforms au dataset "train_subset" via le DatasetWrapper
    # Mais comme dataset.py applique transform au __getitem__, on doit passer les transforms au constructeur.
    # SIMPLIFICATION : On va recharger le dataset en deux parties pour être propre.
    
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = VivaDataset(DATA_DIR, mode=args.network.lower(), indices=train_indices, transform=train_transforms)
    val_data = VivaDataset(DATA_DIR, mode=args.network.lower(), indices=val_indices, transform=val_transforms)

    # 3. DataLoaders
    # Papier: Batch size 40 pour LRN, 20 pour HRN.
    batch_size = 40 if args.network == 'lrn' else 20
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Modèle
    if args.network == 'hrn':
        model = HRN(num_classes=NUM_CLASSES)
    else:
        model = LRN(num_classes=NUM_CLASSES)

    # 5. Entraînement
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=args.network.upper(),
        device=DEVICE
    )
    
    trainer.fit(epochs=args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, required=True, choices=['hrn', 'lrn'], help="Réseau à entraîner (hrn ou lrn)")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques max")
    args = parser.parse_args()
    
    main(args)