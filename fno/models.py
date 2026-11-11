import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .layers import SpectralConv2d
except ImportError:
    from layers import SpectralConv2d


class FNO2d(nn.Module):
    """
    Fourier Neural Operator en 2D.
    
    Parámetros:
    -----------
    modes1 : int
        Número de modos de Fourier a retener en la primera dimensión espacial.
    modes2 : int
        Número de modos de Fourier a retener en la segunda dimensión espacial.
    width : int
        Canales ocultos / dimensión latente del operador.
    in_channels : int, default=1
        Número de variables físicas de entrada.
    out_channels : int, default=1
        Número de variables físicas de salida predichas.
    """
    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        in_channels: int = 1,
        out_channels: int = 1
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # 1. Lifting: proyección al espacio latente (canales in -> width)
        self.p = nn.Conv2d(in_channels, self.width, kernel_size=1)

        # 2. Cuatro capas espectrales convolucionales
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # Bypass lineales locales (Convoluciones 1x1 equivalentes a matrices W)
        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1)

        # 3. Proyección final de vuelta al espacio físico
        self.q1 = nn.Conv2d(self.width, 128, kernel_size=1)
        self.q2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Etapa 1: Lifting
        x = self.p(x)

        # Etapa 2: Bloques Fourier + Conexión residual local + Activación
        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)

        x1 = self.conv1(x) + self.w1(x)
        x = F.gelu(x1)

        x1 = self.conv2(x) + self.w2(x)
        x = F.gelu(x1)

        # El último bloque espectral usualmente no lleva activación intermedia
        x1 = self.conv3(x) + self.w3(x)
        x = x1

        # Etapa 3: Proyección
        x = self.q1(x)
        x = F.gelu(x)
        x = self.q2(x)
        return x