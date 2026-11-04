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

        # Loading everything to RAM
        with h5py.File(self.ruta_hdf5, "r") as f:
            x_aod_full = f["X_aod"][:]
            y_img_full = f["Y_img"][:]
            y_int_full = f["Y_int"][:]

        total_muestras = len(x_aod_full)
        self.indices = np.array(indices) if indices is not None else np.arange(total_muestras)
        self.longitud = len(self.indices)

        # Filter and convert to float32
        self.x_aod_ram = x_aod_full[self.indices].astype(np.float32)
        self.y_img_ram = y_img_full[self.indices].astype(np.float32)
        self.y_int_ram = y_int_full[self.indices].astype(np.float32)

        # Computing maximums
        self.amp_max_global = amp_max if amp_max is not None else np.max(self.x_aod_ram[:, :, 0])
        self.int_max_global = int_max if int_max is not None else np.max(self.y_int_ram)
        
        # log1p to full matrix
        self.y_img_ram = np.log1p(self.y_img_ram)
        self.img_max_global = img_max if img_max is not None else np.max(self.y_img_ram)

        # Vectorization
        
        # AOD: Amplitudes
        self.x_aod_ram[:, :, 0] /= self.amp_max_global
        
        # AOD: Phases
        fases = self.x_aod_ram[:, :, 1]
        fases_wrapped = np.arctan2(np.sin(fases), np.cos(fases))
        self.x_aod_ram[:, :, 1] = fases_wrapped / np.pi

        # IMAGE
        self.y_img_ram /= self.img_max_global

        # INTENSITIES
        self.y_int_ram /= self.int_max_global

    def __len__(self):
        return self.longitud

    def __getitem__(self, idx):
        # Converts tensor and adds channel dimension
        return (
            torch.tensor(self.x_aod_ram[idx], dtype=torch.float32),
            torch.tensor(self.y_img_ram[idx], dtype=torch.float32).unsqueeze(0),
            torch.tensor(self.y_int_ram[idx], dtype=torch.float32),
        )