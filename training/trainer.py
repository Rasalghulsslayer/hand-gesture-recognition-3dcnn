import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, model_name, device):
        self.model = model.to(device).float()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        self.criterion = nn.CrossEntropyLoss()
        # OPTIMISEUR : Adam
        # Learning Rate : 1e-3 (0.001)
        # Weight Decay : 1e-4 (un peu de régularisation pour éviter l'overfit)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.0003, 
            weight_decay=1e-4
        )
        
        # Scheduler personnalisé pour imiter la logique du papier :
        # "reduced it by a factor of 2 if cost ... did not improve by more than 10% in 40 epochs" 
        # ReduceLROnPlateau est l'équivalent PyTorch le plus proche.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=40, 
            threshold=0.01, # 1% improvement threshold (papier dit 10% mais 1% est plus standard pour converger finement)
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(self.train_loader, desc="Train", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()

            # --- ESPION DE GRADIENT ---
            # On vérifie la couche 1. Si c'est proche de 0, le réseau est mort.
            grad_norm = self.model.conv1.weight.grad.norm().item()
            if grad_norm < 0.0001:
                 print(f"⚠️ ALERTE : Gradient presque nul ({grad_norm:.6f}) ! Le réseau n'apprend pas.")
            # --------------------------

            self.optimizer.step()
            
            # Stats
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def fit(self, epochs=300): # 
        best_acc = 0.0
        save_path = Path("checkpoints")
        save_path.mkdir(exist_ok=True)
        
        print(f"🚀 Démarrage de l'entraînement pour {self.model_name}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Mise à jour du learning rate
            self.scheduler.step(val_loss)
            
            # Checkpoint du meilleur modèle
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path / f"best_{self.model_name}.pth")
                print(f"💾 Nouveau record ! Modèle sauvegardé (Acc: {val_acc:.2f}%)")
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
            
            # Stop condition logic du papier (si LR trop bas)
            # current_lr = self.optimizer.param_groups[0]['lr']
            #if current_lr < 0.005 / (2**4): # "decayed at least 4 times" 
            #   print("🛑 Arrêt anticipé : Learning Rate trop bas.")
            #    break