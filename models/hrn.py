import torch
import torch.nn as nn

class HRN(nn.Module):
    def __init__(self, num_classes=34, in_channels=1):
        super(HRN, self).__init__()
        
        # Layer 1 [cite: 93, 100]
        # Conv: 4 filtres, kernel 7x7x5
        self.conv1 = nn.Conv3d(in_channels, 4, kernel_size=(5, 7, 7), padding=(2, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)) # [cite: 100]
        self.relu1 = nn.ReLU()

        # Layer 2 [cite: 95, 101]
        self.conv2 = nn.Conv3d(4, 8, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.relu2 = nn.ReLU()

        # Layer 3 [cite: 98, 102]
        self.conv3 = nn.Conv3d(8, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 1)) # Ajusté pour éviter dimension 0
        self.relu3 = nn.ReLU()

        # Layer 4 [cite: 103, 104]
        self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 3, 5), padding=(1, 1, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 1)) 
        self.relu4 = nn.ReLU()

        # Fully Connected Layers [cite: 119-123]
        self.dropout3d = nn.Dropout3d(p=0.5) # Pour les Conv3D
        self.dropout = nn.Dropout(p=0.5)     # Pour les Linear (FC)
        
        # Calcul dynamique de la taille aplatie pour éviter les erreurs
        self._to_linear = None
        self._get_flatten_size((1, in_channels, 32, 57, 125))

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes) # Sortie logits (34 classes)

        self._init_weights()

    def _get_flatten_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(shape)
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            x = self.pool4(self.relu4(self.conv4(x)))
            self._to_linear = x.view(1, -1).size(1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # KAIMING INITIALIZATION (Le défibrillateur pour Conv3D)
                # Mode fan_out préserve la variance dans la passe arrière
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Exception pour la dernière couche
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = self.relu4(self.pool4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu1(self.fc1(x)) # Réutilisation de relu1 par commodité
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x