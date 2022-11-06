from torch import nn


h1=128
h2=64
h3=16
h4=1
dim=3

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim,h1), nn.ReLU(True),
            nn.Linear(h1, h2),
            nn.ReLU(True),
            nn.Linear(h2, h3),
            nn.ReLU(True), nn.Linear(h3,1))

    def forward(self, x):
        x = self.encoder(x)
        return x
        
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1, h4),
            nn.ReLU(True),
            nn.Linear(h4, h3),
            nn.ReLU(True),
            nn.Linear(h3, h2),
            nn.ReLU(True),
            nn.Linear(h2, h1),
            nn.ReLU(True), nn.Linear(h1, dim), nn.Tanh())

    def forward(self, x):
        x = self.decoder(x)
        return x
