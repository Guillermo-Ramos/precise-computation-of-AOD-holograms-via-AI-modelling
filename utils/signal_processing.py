import numpy as np
import torch


def integrar_picos_robusto(img, approx_x, rango_y=(16, 20), margen_x=4, ventana=5):
    """
    img: La imagen cruda de la cámara.
    approx_x: Lista de las 38 posiciones X aproximadas.
    rango_y: Tupla (fila_min, fila_max) donde sabes que SIEMPRE cae la línea del AOD.
    margen_x: Píxeles a izquierda y derecha permitidos para corregir derivas.
    ventana: Tamaño del cuadrado de integración (5x5).
    """
    intensidades = np.zeros(len(approx_x))
    offset = ventana // 2
    
    # Restricted y search
    y_min, y_max = rango_y
    zona_y = img[y_min:y_max, :] 
    
    # Encontramos la fila con más luz SOLO dentro de esa banda
    perfil_y = np.sum(zona_y, axis=1)
    y_centro = y_min + np.argmax(perfil_y) # Sumamos y_min para volver a la coordenada global
    
    # 2. BÚSQUEDA LOCAL EN X PARA CADA PICO
    for i, x_aprox in enumerate(approx_x):
        # Delimitamos la "caja" de búsqueda horizontal
        x_ini_busq = max(0, x_aprox - margen_x)
        x_fin_busq = min(img.shape[1], x_aprox + margen_x + 1)
        
        # Tomamos una pequeña zona (1 o 2 píxeles de alto) centrada en y_centro
        zona_local_x = img[max(0, y_centro-1) : min(img.shape[0], y_centro+2), 
                                      x_ini_busq : x_fin_busq]
        
        # Encontramos el pico exacto DENTRO de esa caja
        perfil_x_local = np.sum(zona_local_x, axis=0)
        x_real = x_ini_busq + np.argmax(perfil_x_local)
        
        # 3. INTEGRACIÓN EN EL CUADRADO EXACTO (Ventana 5x5)
        y_ini_int = max(0, y_centro - offset)
        y_fin_int = min(img.shape[0], y_centro + offset + 1)
        x_ini_int = max(0, x_real - offset)
        x_fin_int = min(img.shape[1], x_real + offset + 1)
        
        recorte = img[y_ini_int:y_fin_int, x_ini_int:x_fin_int]
        intensidades[i] = np.sum(recorte)
        
    return intensidades

def normalize_log_minmax(img, eps=1e-8):
    '''
    Comprises dynamic range with log1p and scales to [0,1).
    '''
    if isinstance(img, np.ndarray):
        img_log = np.log1p(np.maximum(0, img))
        max_val = np.max(img_log)
        return img_log / (max_val + eps)
    else:
        img_log = torch.log1p(torch.clamp(img, min=0))
        max_val = torch.max(img_log)
        return img_log / (max_val + eps)

def denormalize_log_minmax(img_norm, max_val_original):
    '''
    Inverts log1p.
    '''
    if isinstance(img_norm, np.ndarray):
        img_log = img_norm * max_val_original
        return np.expm1(img_log)
    else:
        img_log = img_norm * max_val_original
        return torch.expm1(img_log)

def normalize_aod(amplitude, phase, amp_max=1.0):
    '''
    Normalizes amplitudes to [0,1] and converts phases to sin/cos to avoid discontinuities on periodicity.
    '''
    amp_norm = amplitude / amp_max
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)

    # Returns array (38,3)
    return np.column_stack((amp_norm, sin_phase, cos_phase))