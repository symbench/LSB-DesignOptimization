from torch import nn


class encoder(nn.Module):
    def __init__(self,h1,h2,h3,h4,z):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(z,h1), nn.ReLU(True),
            nn.Linear(h1, h2),
            nn.ReLU(True),
            nn.Linear(h2, h3),
            nn.ReLU(True), nn.Linear(h3,h4))

    def forward(self, x):
        x = self.encoder(x)
        return x
        
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(h4, h3),
            nn.ReLU(True),
            nn.Linear(h3, h2),
            nn.ReLU(True),
            nn.Linear(h2, h1),
            nn.ReLU(True), nn.Linear(h1, z), nn.Tanh())

    def forward(self, x):
        x = self.decoder(x)
        return x
