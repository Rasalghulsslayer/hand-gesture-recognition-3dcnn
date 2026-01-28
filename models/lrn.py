import torch
import torch.nn as nn

class LRN(nn.Module):
    def __init__(self, num_classes=34, in_channels=1):
        super(LRN, self).__init__()
        
        # Layer 1 [cite: 108, 129]
        self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=(5, 5, 5), padding=(2, 2, 2))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.relu1 = nn.ReLU()

        # Layer 2 [cite: 113, 130]
        self.conv2 = nn.Conv3d(8, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.relu2 = nn.ReLU()

        # Layer 3 [cite: 117, 131]
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 5), padding=(1, 1, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 4, 1))
        self.relu3 = nn.ReLU()

        self.dropout3d = nn.Dropout3d(p=0.5) # Pour les Conv3D
        self.dropout = nn.Dropout(p=0.5)     # Pour les Linear (FC)
        
        # Calcul automatique de la taille pour le Flatten
        self._to_linear = None
        self._get_flatten_size((1, in_channels, 32, 28, 62))
        
        # Fully Connected [cite: 133-135]
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self._init_weights()

    def _get_flatten_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(shape)
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            self._to_linear = x.view(1, -1).size(1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 1)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x) 
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x