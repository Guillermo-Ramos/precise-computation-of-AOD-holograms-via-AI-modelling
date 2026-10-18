import torch
import torch.nn as nn
import torch.optim as optim

class AODtoHologramCNN(nn.Module):
    def __init__(self, input_dim=2, output_size=128):
        super(AODtoHologramCNN, self).__init__()
        
        # 1. Capa Densa (Fully Connected)
        # Proyectamos los 2 números de entrada a un vector lo suficientemente 
        # grande para formar una imagen base (ej. 128 canales de 16x16 píxeles)
        self.fc = nn.Linear(input_dim, 128 * 16 * 16)
        
        # 2. Capas de Convolución Transpuesta (Upsampling)
        self.decoder = nn.Sequential(
            # Entrada: 128 canales x 16 x 16 -> Salida: 64 canales x 32 x 32
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Entrada: 64 canales x 32 x 32 -> Salida: 32 canales x 64 x 64
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Entrada: 32 canales x 64 x 64 -> Salida: 1 canal (escala de grises) x 128 x 128
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
            
            # Usamos Sigmoid porque queremos que los píxeles de salida estén entre 0 y 1 (intensidad)
            nn.Sigmoid()
        )

    def forward(self, x):
        # x tiene tamaño: (batch_size, 2)
        x = self.fc(x)
        
        # Remodelamos el vector plano a un tensor 4D: (batch_size, canales, alto, ancho)
        x = x.view(-1, 128, 16, 16) 
        
        # Pasamos el tensor por las capas convolucionales
        x = self.decoder(x)
        return x

# ==========================================
# Ejemplo de uso e inicialización
# ==========================================

# Instanciamos el modelo
# Si decides incluir la Fase del AOD además de Frecuencia y Amplitud, cambia input_dim a 3
modelo = AODtoHologramCNN(input_dim=2)

# Simulamos un lote (batch) de 4 mediciones del AOD
# Formato: [Frecuencia, Amplitud] normalizados entre 0 y 1
datos_aod = torch.tensor([
    [0.50, 0.80], 
    [0.75, 0.90], 
    [0.25, 0.50], 
    [0.90, 0.20]
], dtype=torch.float32)

# Generamos los hologramas
hologramas_predichos = modelo(datos_aod)

print(f"Tamaño de entrada: {datos_aod.shape}") 
print(f"Tamaño de salida de las imágenes: {hologramas_predichos.shape}") 
# Salida esperada: torch.Size([4, 1, 128, 128])




#####################

Programa 2

import numpy as np
import matplotlib.pyplot as plt

def generar_holograma_simulado(frecuencia_aod, amplitud_aod, size=128):
    """
    Genera un holograma 2D basado en la frecuencia y amplitud del AOD.
    Los valores de entrada deben estar normalizados entre 0 y 1.
    """
    # 1. Crear la cuadrícula de píxeles (plano XY)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 2. Mapear frecuencia AOD a frecuencia espacial (densidad de las franjas)
    # Multiplicamos por un factor (ej. 20) para que se vean varias franjas en la imagen
    k = frecuencia_aod * 20  
    
    # 3. Calcular el patrón de interferencia sinusoidal
    fase = 0 # Asumimos fase 0, pero podrías hacerla aleatoria
    patron_franjas = 0.5 * (1 + np.cos(2 * np.pi * k * X + fase))
    
    # 4. Aplicar el perfil Gaussiano del láser (más brillante en el centro)
    sigma = 0.5 # Anchura del rayo
    perfil_gaussiano = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # 5. Combinar todo y escalar por la amplitud del AOD
    holograma_ideal = amplitud_aod * patron_franjas * perfil_gaussiano
    
    # 6. Añadir ruido de disparo (ruido de Poisson/Gaussiano) típico de las cámaras
    ruido = np.random.normal(0, 0.05, holograma_ideal.shape) # 5% de ruido
    holograma_realista = np.clip(holograma_ideal + ruido, 0, 1) # Mantener valores entre 0 y 1
    
    return holograma_realista

# ==========================================
# Script para generar el Dataset de Entrenamiento
# ==========================================

num_muestras = 5000
tamano_imagen = 128

# Arrays vacíos para guardar datos (X: Entradas, Y: Imágenes)
datos_X = np.zeros((num_muestras, 2))
datos_Y = np.zeros((num_muestras, 1, tamano_imagen, tamano_imagen))

for i in range(num_muestras):
    # Generar parámetros aleatorios normalizados entre 0 y 1
    # Frecuencia mínima > 0.1 para evitar imágenes vacías o sin franjas
    f_aod = np.random.uniform(0.1, 1.0) 
    a_aod = np.random.uniform(0.2, 1.0) 
    
    # Generar la imagen correspondiente
    img = generar_holograma_simulado(f_aod, a_aod, size=tamano_imagen)
    
    # Guardar en el dataset
    datos_X[i] = [f_aod, a_aod]
    datos_Y[i, 0] = img # El '0' es por el canal de color (1 canal = escala de grises)

print(f"Dataset generado con éxito.")
print(f"Formato de X (Frecuencia, Amplitud): {datos_X.shape}")
print(f"Formato de Y (Hologramas): {datos_Y.shape}")

# Visualizar un ejemplo
plt.imshow(datos_Y[0, 0], cmap='gray')
plt.title(f"Freq: {datos_X[0, 0]:.2f} | Amp: {datos_X[0, 1]:.2f}")
plt.colorbar()
plt.show()