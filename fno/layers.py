import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    """
    Capa de convolución espectral 2D para Fourier Neural Operators.
    Aplica una transformada rápida de Fourier (FFT), multiplica los modos
    de baja frecuencia por un tensor de pesos complejos aprendibles y
    regresa al espacio físico mediante la transformada inversa (iFFT).
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Cantidad de modos conservados en el eje Y (dim -2)
        self.modes2 = modes2  # Cantidad de modos conservados en el eje X (dim -1)

        # Factor de escala para que la inicialización gaussiana tenga varianza controlada
        scale = 1.0 / (in_channels * out_channels)

        # Pesos complejos aprendibles para las esquinas de frecuencias bajas (superior e inferior)
        self.weights1 = nn.Parameter(
            scale * torch.view_as_complex(torch.randn(in_channels, out_channels, modes1, modes2, 2))
        )
        self.weights2 = nn.Parameter(
            scale * torch.view_as_complex(torch.randn(in_channels, out_channels, modes1, modes2, 2))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, in_channels, H, W]
        batch_size = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # 1. Transformada rápida de Fourier 2D para señales reales
        # Pasa del dominio físico R^[H, W] al dominio de frecuencias C^[H, W//2 + 1]
        x_ft = torch.fft.rfft2(x, norm="forward")

        # 2. Lienzo de ceros en espacio de Fourier para la salida
        out_ft = torch.zeros(
            batch_size, self.out_channels, H, W // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        # 3. Multiplicación espectral de bajas frecuencias superiores
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )

        # 4. Multiplicación espectral de bajas frecuencias inferiores (modos conjugados negativos)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        # 5. Transformada inversa de Fourier al dominio físico
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="forward")
        return x_out