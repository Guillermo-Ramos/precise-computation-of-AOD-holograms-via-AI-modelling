import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetMicroscope(Dataset):

    def __init__(
        self,
        ruta_hdf5,
        indices=None,
        amp_max=None,
        img_max=None,
        int_max=None,
    ):
        self.ruta_hdf5 = ruta_hdf5

        # 1. CARGA COMPLETA A RAM
        with h5py.File(self.ruta_hdf5, "r") as f:
            x_aod_full = f["X_aod"][:]
            y_img_full = f["Y_img"][:]
            y_int_full = f["Y_int"][:]

        total_muestras = len(x_aod_full)

        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = np.arange(total_muestras)

        self.longitud = len(self.indices)

        # Filtrar muestras en RAM
        self.x_aod_ram = x_aod_full[self.indices]
        self.y_img_ram = y_img_full[self.indices]
        self.y_int_ram = y_int_full[self.indices]

        # 2. CÁLCULO DINÁMICO DE MÁXIMOS SINO SE PROPORCIONAN
        if amp_max is None:
            # Saca el máximo real exacto de la Columna 0 (Amplitudes) de este dataset
            self.amp_max_global = np.max(self.x_aod_ram[:, :, 0])
        else:
            self.amp_max_global = amp_max

        if int_max is None:
            self.int_max_global = np.max(self.y_int_ram)
        else:
            self.int_max_global = int_max

        if img_max is None:
            img_log = np.log1p(self.y_img_ram.astype(np.float32))
            self.img_max_global = np.max(img_log)
        else:
            self.img_max_global = img_max

    def __len__(self):
        return self.longitud

    def __getitem__(self, idx):
        aod_raw = self.x_aod_ram[idx]
        img_raw = self.y_img_ram[idx]
        y_int_raw = self.y_int_ram[idx]

        # --- AOD ---
        amp = aod_raw[:, 0]
        fase = aod_raw[:, 1]

        # 1. Amplitud: División dinámica por el máximo real (sin np.clip, sin hardcode)
        amp_norm = amp / self.amp_max_global  # Queda en [0, 1]

        # 2. Fase: Empaquetado periódico a [-pi, pi] y escalado a [-1, 1]
        fase_wrapped = np.arctan2(np.sin(fase), np.cos(fase))
        fase_norm = fase_wrapped / np.pi

        aod_proc = np.column_stack((amp_norm, fase_norm))

        # --- IMAGEN ---
        img_log = np.log1p(img_raw.astype(np.float32))
        img_norm = img_log / self.img_max_global

        # --- INTENSIDADES ---
        y_int_norm = y_int_raw / self.int_max_global

        return (
            torch.tensor(aod_proc, dtype=torch.float32),
            torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_int_norm, dtype=torch.float32),
        )