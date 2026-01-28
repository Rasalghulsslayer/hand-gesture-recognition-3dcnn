import torch
from torch.utils.data import DataLoader
from training.dataset import VivaDataset
from training.augmentations import Compose, ToTensor
from models.hrn import HRN
from models.lrn import LRN
# On importe ta fonction de fusion
from models.fusion import fuse_predictions
import sys
import os

# --- CONFIGURATION ---
DATA_DIR = "data/processed"
BATCH_SIZE = 32
NUM_CLASSES = 34
# Détection automatique : Mac (MPS), Nvidia (CUDA) ou CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"--- 🚀 ÉVALUATION FINALE (FUSION) sur {DEVICE} ---")
    
    # 1. Chargement des Données
    # Important : Pas d'augmentation (Random) ici, on veut des résultats stables.
    transforms = Compose([ToTensor()])
    
    print("Chargement du dataset...")
    # On crée deux instances du dataset. 
    # Grâce à ton script de réparation, hrn_0.pt et lrn_0.pt correspondent bien au même geste.
    ds_hrn = VivaDataset(DATA_DIR, mode='hrn', transform=transforms)
    ds_lrn = VivaDataset(DATA_DIR, mode='lrn', transform=transforms)
    
    if len(ds_hrn) != len(ds_lrn):
        print("❌ Erreur critique : Les datasets HRN et LRN n'ont pas la même taille !")
        return

    # CRUCIAL : shuffle=False pour que les deux chargeurs envoient les mêmes vidéos en même temps
    loader_hrn = DataLoader(ds_hrn, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    loader_lrn = DataLoader(ds_lrn, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Chargement des Modèles
    print("Chargement des poids des modèles...")
    net_hrn = HRN(num_classes=NUM_CLASSES).to(DEVICE).float()
    net_lrn = LRN(num_classes=NUM_CLASSES).to(DEVICE).float()
    
    path_hrn = "checkpoints/best_HRN.pth"
    path_lrn = "checkpoints/best_LRN.pth"

    if not os.path.exists(path_hrn) or not os.path.exists(path_lrn):
        print(f"❌ Erreur : Fichiers de poids introuvables.")
        print(f"   Cherché ici : {path_hrn} et {path_lrn}")
        return

    net_hrn.load_state_dict(torch.load(path_hrn, map_location=DEVICE))
    net_lrn.load_state_dict(torch.load(path_lrn, map_location=DEVICE))
    
    net_hrn.eval()
    net_lrn.eval()
    print("✅ Modèles chargés et prêts.")

    # 3. Boucle d'évaluation conjointe
    correct_hrn = 0
    correct_lrn = 0
    correct_fusion = 0
    total = 0
    
    print(f"Calcul en cours sur {len(ds_hrn)} échantillons...")
    
    with torch.no_grad():
        # 'zip' permet de parcourir les deux DataLoaders simultanément
        for i, ((x_h, labels), (x_l, _)) in enumerate(zip(loader_hrn, loader_lrn)):
            
            x_h, x_l, labels = x_h.to(DEVICE), x_l.to(DEVICE), labels.to(DEVICE)
            
            # --- A. Prédictions Brutes (Logits) ---
            logits_h = net_hrn(x_h)
            logits_l = net_lrn(x_l)
            
            # --- B. Évaluation HRN Seul ---
            _, pred_h = logits_h.max(1)
            correct_hrn += pred_h.eq(labels).sum().item()
            
            # --- C. Évaluation LRN Seul ---
            _, pred_l = logits_l.max(1)
            correct_lrn += pred_l.eq(labels).sum().item()
            
            # --- D. FUSION (Ta fonction) ---
            # fuse_predictions prend les logits et renvoie les probabilités fusionnées
            probs_final = fuse_predictions(logits_h, logits_l)
            
            # On prend l'indice de la probabilité maximale
            _, pred_fusion = probs_final.max(1)
            correct_fusion += pred_fusion.eq(labels).sum().item()
            
            total += labels.size(0)
            
            # Affichage progression
            if i % 5 == 0:
                sys.stdout.write(f"\rBatch {i+1}/{len(loader_hrn)} traité...")
                sys.stdout.flush()

    # 4. Calcul des pourcentages
    acc_hrn = 100.0 * correct_hrn / total
    acc_lrn = 100.0 * correct_lrn / total
    acc_fusion = 100.0 * correct_fusion / total
    
    print(f"\n\n🏆 RÉSULTATS FINAUX (Sur {total} gestes) 🏆")
    print("="*45)
    print(f"🔹 LRN (Basse Résolution) : {acc_lrn:.2f}%")
    print(f"🔹 HRN (Haute Résolution) : {acc_hrn:.2f}%")
    print("-" * 45)
    print(f"🚀 FUSION COMBINÉE        : {acc_fusion:.2f}%")
    print("="*45)
    
    # Petit diagnostic final
    gain = acc_fusion - max(acc_hrn, acc_lrn)
    if gain > 0:
        print(f"✅ La fusion a apporté un gain de +{gain:.2f}% !")
    else:
        print("⚠️ La fusion n'a pas amélioré le score (vérifier si un modèle domine l'autre).")

if __name__ == "__main__":
    evaluate()