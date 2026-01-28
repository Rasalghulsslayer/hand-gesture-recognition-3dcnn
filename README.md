# hand-gesture-recognition-3dcnn
Making the network illustrated in the Hand Gesture Recognition with 3D Convolutional Neural Networks Pavlo Molchanov, Shalini Gupta, Kihwan Kim, and Jan Kautz paper for education purposes.

---

# 🖐️ Reconnaissance de Gestes de la Main par 3D CNN

Ce projet est une implémentation PyTorch de l'architecture de reconnaissance de gestes décrite dans le papier **"Hand Gesture Recognition with 3D Convolutional Neural Networks"** (Molchanov et al., 2015).

Le système utilise une approche à deux flux (**Two-Stream Network**) pour classer 34 gestes dynamiques de la main (VIVA Challenge) :

1. **LRN (Low Resolution Network) :** Analyse globale du mouvement.
2. **HRN (High Resolution Network) :** Analyse détaillée de la forme de la main.
3. **Fusion :** Combinaison des deux réseaux pour la prédiction finale.

---

## 📂 Structure du Projet

```bash
.
├── data/
│   ├── raw/                # Données brutes (.npy) + labels.npy
│   └── processed/          # Tenseurs PyTorch prêts à l'emploi (.pt)
├── models/
│   ├── lrn.py              # Architecture Basse Résolution
│   ├── hrn.py              # Architecture Haute Résolution
│   └── fusion.py           # Logique de fusion des probabilités
├── training/
│   ├── dataset.py          # Dataset Loader (gère le chargement des données)
│   ├── augmentations.py    # Augmentations Spatiales & Temporelles (TED)
│   └── trainer.py          # Boucle d'entraînement (Adam/SGD)
├── checkpoints/            # Sauvegarde des meilleurs poids (.pth)
├── preprocessing/
│   └── fix_dataset_ordering.py  # Script de reconstruction du dataset
├── train_net.py            # Script principal d'entraînement
├── evaluate_fusion.py      # Script d'évaluation finale
└── README.md

```

---

## ⚙️ Installation

Assurez-vous d'avoir Python installé (3.8+ recommandé). Installez les dépendances :

```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm

```

*Note : Le code détecte automatiquement l'accélération matérielle (CUDA pour Nvidia ou MPS pour Mac Silicon).*

---

## 🚀 Guide d'Utilisation

### 1. Préparation des Données

Les données brutes doivent être placées dans `data/raw/` (fichiers `.npy` et `labels.npy`).
Pour garantir l'alignement entre les vidéos et les labels, et générer les tenseurs PyTorch normalisés :

```bash
python preprocessing/fix_dataset_ordering.py

```

*Cela va générer des milliers de fichiers `.pt` (LRN et HRN) dans `data/processed/`.*

### 2. Entraînement des Réseaux

Les deux réseaux s'entraînent séparément. Le script applique automatiquement des augmentations de données (Rotation, Scale, Déformation Temporelle Élastique) pour éviter l'overfitting.

**Entraîner le LRN (Low Resolution) :**

```bash
python train_net.py --network lrn --epochs 50

```

**Entraîner le HRN (High Resolution) :**

```bash
python train_net.py --network hrn --epochs 50

```

Les meilleurs modèles (basés sur l'accuracy de validation) seront sauvegardés automatiquement :

* `checkpoints/best_LRN.pth`
* `checkpoints/best_HRN.pth`

### 3. Évaluation Finale (Fusion)

Une fois les deux réseaux entraînés, lancez le script de fusion. Il va combiner les probabilités des deux modèles (multiplication élément par élément) pour produire la décision finale.

```bash
python evaluate_fusion.py

```

Le script affichera les précisions comparées :

* Score LRN seul
* Score HRN seul
* **Score Fusionné (Résultat final)**

---

## 🧠 Détails Techniques

### Pipeline de Données

* **Input :** Vidéo de profondeur + Gradient (N frames).
* **Preprocessing :**
* Re-échantillonnage temporel à **32 frames**.
* Normalisation (Standard Score).
* Resize :  (LRN) et  (HRN).


* **Augmentations (Online) :**
* Augmentations spatiales affines (Rotation , Scale ).
* **Temporal Elastic Deformation (TED) :** Déformation temporelle locale pour simuler des variations de vitesse d'exécution du geste.



### Hyperparamètres

* **Optimiseur :** Adam (LR = 1e-3) ou SGD (LR = 0.005, Momentum=0.9, Nesterov).
* **Scheduler :** ReduceLROnPlateau (divise le LR par 2 si stagnation).
* **Initialisation :** Kaiming He (adaptée pour les activations ReLU).
* **Dropout :** Appliqué après les couches linéaires pour la régularisation.

---

## 📊 Résultats Attendus

Sur le dataset VIVA (34 classes), l'architecture typique donne les ordres de grandeur suivants :

* LRN : ~40-50%
* HRN : ~60-70%
* **Fusion : ~+3% à +5% par rapport au meilleur modèle seul.**

---

## 📝 Référence

> Molchanov, P., Gupta, S., Kim, K., & Kautz, J. (2015). "Hand gesture recognition with 3D convolutional neural networks". In Proceedings of the IEEE conference on computer vision and pattern recognition workshops.