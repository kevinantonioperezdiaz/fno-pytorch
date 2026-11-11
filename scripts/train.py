import time
import os
import sys
from pathlib import Path

# 1. Asegurar que Python vea la raíz del proyecto (fno-pytorch)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 2. Ahora sí importamos librerías y nuestro módulo fno
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_navier_stokes_pt

from fno import FNO2d, RelL2Loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Usando dispositivo: {device}")

    print("-> Descargando/Cargando dataset de Navier-Stokes...")
    train_loader, test_loaders, _ = load_navier_stokes_pt(
        n_train=100,
        n_tests=[20],
        batch_size=10,
        test_batch_sizes=[10],
        train_resolution=128,
        test_resolutions=[128]
    )

    model = FNO2d(modes1=12, modes2=12, width=32, in_channels=1, out_channels=1).to(device)
    criterion = RelL2Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Parametros entrenables: {total_params:,}")

    epochs = 10
    print(f"-> Iniciando entrenamiento por {epochs} epocas...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 2 == 0 or epoch == 1:
            print(f"   Epoca [{epoch:02d}/{epochs}] - Loss Rel L2: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"-> Entrenamiento completado en {total_time:.2f} segundos.")

    model.eval()
    test_loader_128 = test_loaders[128]
    test_batch = next(iter(test_loader_128))
    x_test = test_batch["x"].to(device)
    y_test = test_batch["y"].to(device)

    with torch.no_grad():
        y_pred = model(x_test)
        test_loss = criterion(y_pred, y_test)

    print(f"-> Error Relativo L2 en test (res 128x128): {test_loss.item():.4f}")

    print("-> Generando grafica de prediccion...")
    idx = 0
    entrada = x_test[idx, 0].cpu().numpy()
    verdad = y_test[idx, 0].cpu().numpy()
    prediccion = y_pred[idx, 0].cpu().numpy()
    error_abs = np.abs(verdad - prediccion)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    im0 = axes[0].imshow(entrada, cmap="twilight")
    axes[0].set_title(r"1. Entrada Inicial ($t_0$)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(verdad, cmap="twilight")
    axes[1].set_title(r"2. Ground Truth ($t_{final}$)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(prediccion, cmap="twilight")
    axes[2].set_title("3. Prediccion FNO")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(error_abs, cmap="inferno")
    axes[3].set_title("4. Error Absoluto")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path = os.path.join(ROOT_DIR, "fno_navier_stokes_result.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Grafica guardada con exito en: {output_path}")

if __name__ == "__main__":
    main()