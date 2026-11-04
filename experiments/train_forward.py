import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Configuring paths to find src/ y utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import DatasetMicroscope
from src.models.forward_model import DirectModelAOD


DATA_PATH = PROJECT_ROOT / "data" / "real" / "processed" / "training_dataset.h5"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "direct_model.pth"
WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)


# Hyperparameters

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 5e-4

# Hardware selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Training on NVIDIA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Training on Apple Silicon (MPS)")
else:
    device = torch.device("cpu")
    print("Training on CPU")

torch.set_float32_matmul_precision('high') # Activa TF32

# Data preparation
print("\n[1] Loading data an dividing Train / Valid...")
total_samples = 4000
train_size = int(0.8 * total_samples)

np.random.seed(10)
mixed_indices = np.random.permutation(total_samples)
train_indices = mixed_indices[:train_size]
val_indices = mixed_indices[train_size:]

# Train Dataset: Global maximums
train_dataset = DatasetMicroscope(DATA_PATH, indices=train_indices)

# Valid Dataset: Uses Global maximums from Train, to avoid data leakage
val_dataset = DatasetMicroscope(
    DATA_PATH, 
    indices=val_indices,
    amp_max=train_dataset.amp_max_global,
    img_max=train_dataset.img_max_global,
    int_max=train_dataset.int_max_global
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# MODEL PREPARATION
print("\n[2] Initiating model, loss function and optimizer...")
model = DirectModelAOD().to(device)
model = torch.compile(model)

# POR CAMBIAR
'''
Una opcion es tener en cuenta que la intensidad de luz total debe ser la misma. Conservación energia
'''
def criterio_ponderado(prediction, objetivo):
    #error al cuadrado estandar
    mse_base = (prediction - objetivo) **2
    # mascara que vale mas donde la imagen real tiene luz (picos)
    #objetivo va de 0 a 1. Hacemos que los picos penalicen hasta 10 veces más.
    factor_importancia = 1.0 + (9.0 * objetivo)
    #multiplicamos el error por la imporancia y sacamos la media
    loss_final = torch.mean(mse_base * factor_importancia)
    return loss_final

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE) 

best_loss_val = float('inf')

# TRAINING LOOP
print(f"\n[3] Initiating training for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    # TRAINING
    model.train()
    loss_train_total = 0.0
    scaler = torch.cuda.amp.GradScaler() # Solo estrictamente necesario para float16, pero buena práctica
    for x_aod, y_img, _ in train_loader:
        # Sends data to GPU
        x_aod, y_img = x_aod.to(device), y_img.to(device)
        y_img = y_img.to(device)
        
        # Gradients to zero
        optimizer.zero_grad(set_to_none=True) # faster than .zero_grad() 
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Forward pass
            prediction = model(x_aod)
            # Computes loss
            loss = criterio_ponderado(prediction, y_img)
        
        # Backward pass - Derivatives computation
        scaler.scale(loss).backward()
        # Updating model weights
        scaler.step(optimizer)
        scaler.update()
        
        loss_train_total += loss.item() * x_aod.size(0)
        
    mean_loss_train = loss_train_total / len(train_dataset)
    
    # EVALUATION
    model.eval() # Evaluation mode, so no Dropout, BatchNorm, ...
    loss_val_total = 0.0
    
    with torch.no_grad():
        for x_aod, y_img, _ in val_loader:
            x_aod = x_aod.to(device)
            y_img = y_img.to(device)
            
            prediction = model(x_aod)
            loss = criterio_ponderado(prediction, y_img)
            
            loss_val_total += loss.item() * x_aod.size(0)
            
    mean_loss_val = loss_val_total / len(val_dataset)
    
    # Saving best model
    is_it_saved = ""
    if mean_loss_val < best_loss_val:
        best_loss_val = mean_loss_val
        torch.save(model.state_dict(), WEIGHTS_PATH)
        is_it_saved = "Saved model"

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {mean_loss_train:.5f} | Val Loss: {mean_loss_val:.5f} {is_it_saved}")

print("\nCompleted")