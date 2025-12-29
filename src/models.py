import torch
from torch import nn


###########################################################################################################
###############                 Klasyczna inplemtacja VNeta (dodana dla refernecji) #######################
###########################################################################################################
class _ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class _UpConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=2, stride=2, padding=0):
        super().__init__()
        self.convt = nn.ConvTranspose3d(in_ch, out_ch, k_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.bn(self.convt(x)))


class _ResidualBlock3d(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm3d(channels))
            layers.append(nn.PReLU())
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.blocks(x)


class _Encoder3d(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 16, num_levels: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = base_channels

        self.layers.append(
            nn.Sequential(
                _ConvBlock3D(in_channels, channels, stride=1),
                _ResidualBlock3d(channels, num_layers=1)
            )
        )

        for level in range(1, num_levels):
            next_channels = channels * 2
            self.layers.append(
                nn.Sequential(
                    _ConvBlock3D(channels, next_channels, stride=2),
                    _ResidualBlock3d(next_channels, num_layers=level + 1)
                )
            )
            channels = next_channels

        self.out_channels = channels  

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                skips.append(x)
        return x, skips


class _Decoder3D(nn.Module):
    def __init__(self, in_channel: int, base_channels: int = 16, num_levels: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channel

        for level in range(num_levels - 1, 0, -1):
            next_channel = channels // 2
            self.layers.append(
                nn.Sequential(
                    _UpConvBlock3D(channels, next_channel),
                    _ResidualBlock3d(next_channel, num_layers=level)
                )
            )
            channels = next_channel

        self.final_conv = nn.Conv3d(channels, base_channels, kernel_size=1)

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(skips):
                x = x + skips[-(i + 1)]
        return self.final_conv(x)
    
class VNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels=16):
        super().__init__()
        self.encoder = _Encoder3d(in_channels=in_ch, base_channels=base_channels)
        self.decoder = _Decoder3D(in_channel=self.encoder.out_channels, base_channels=base_channels)
        self.output_conv = nn.Conv3d(base_channels, out_ch, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.output_conv(x)
        return x 
    
class VNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels=16):
        super().__init__()
        self.encoder = _Encoder3d(in_channels=in_ch, base_channels=base_channels)
        self.decoder = _Decoder3D(in_channel=self.encoder.out_channels, base_channels=base_channels)
        self.output_conv = nn.Conv3d(base_channels, out_ch, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.output_conv(x)
        return x
###########################################################################################################


###########################################################################################################
###############             Ulepsza wersja VNeta z dodnym elemnetem atencji dla skip connect ##############
###########################################################################################################

class _AttentionGate3D(nn.Module):
    """
    Mechanizm atencji filtrujący cechy z enkodera przed ich zsumowaniem w dekoderze.

    Działa to tak, że mamy dwie wastwy atencji:
        - W_g
        - W_x
    W_g widzi tylko glebsze wastwy. Wie ze tu cos jest bo ma duzo cech ale ma niska rodzielczosc. 
    W_x widzi nie wie czy cos jest tu waznego ale widzi to w duzej rodzielczosci

    funkja relu zwraca tylko miejsca gdzie te obydiwe warstwy cos widzą (gdzie sie zgadzają)
    Wyjscie jest potem skalowane przez Sigmoida do postaci amski wag
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # W_g: Sygnał z głębszej warstwy dekodera (gating)
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        # W_x: Sygnał ze skip connection (lokalne cechy)
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        # psi: Wyliczenie wag atencji (0-1)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class _AttEncoder3d(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 16, num_levels: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = base_channels

        self.layers.append(
            nn.Sequential(
                _ConvBlock3D(in_channels, channels, stride=1),
                _ResidualBlock3d(channels, num_layers=1)
            )
        )

        for level in range(1, num_levels):
            next_channels = channels * 2
            self.layers.append(
                nn.Sequential(
                    _ConvBlock3D(channels, next_channels, stride=2),
                    _ResidualBlock3d(next_channels, num_layers=level + 1)
                )
            )
            channels = next_channels

        self.out_channels = channels  

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                skips.append(x)
        return x, skips
    

class _AttDecoder3D(nn.Module):
    def __init__(self, in_channel: int, base_channels: int = 16, num_levels: int = 4):
        super().__init__()
        self.up_layers = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        
        channels = in_channel

        for level in range(num_levels - 1, 0, -1):
            next_channel = channels // 2
            
            self.up_layers.append(
                nn.Sequential(
                    _UpConvBlock3D(channels, next_channel),
                    _ResidualBlock3d(next_channel, num_layers=level)
                )
            )
            
            # Bramka atencji dla danego poziomu
            self.att_gates.append(
                _AttentionGate3D(F_g=next_channel, F_l=next_channel, F_int=next_channel // 2)
            )
            
            channels = next_channel

        self.final_conv = nn.Conv3d(channels, base_channels, kernel_size=1)

    def forward(self, x, skips):
        for i, (up_layer, att_gate) in enumerate(zip(self.up_layers, self.att_gates)):
            x = up_layer(x)  # x to sygnał z dekodera (gating)
            
            # Pobieramy odpowiedni skip connection (od końca)
            skip_feat = skips[-(i + 1)]
            
            # Filtrujemy cechy z enkodera przez atencję
            attended_skip = att_gate(g=x, x=skip_feat)
            
            # Sumujemy (klasyczny V-Net, ale z atencją)
            x = x + attended_skip
            
        return self.final_conv(x)
    
class AttentionVNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels=16):
        super().__init__()
        self.encoder = _AttEncoder3d(in_channels=in_ch, base_channels=base_channels)
        self.decoder = _AttDecoder3D(in_channel=self.encoder.out_channels, base_channels=base_channels)
        self.output_conv = nn.Conv3d(base_channels, out_ch, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        x = self.output_conv(x)
        return x

