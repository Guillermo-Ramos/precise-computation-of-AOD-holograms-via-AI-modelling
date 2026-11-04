import h5py
import numpy as np
import scipy.io
import sys
from pathlib import Path

# To make Python see utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.signal_processing import integrar_picos_robusto



SCRIPT_DIR      = Path(__file__).resolve().parent #src
DATA_DIR        = SCRIPT_DIR.parent / "data" / "real" / "raw"
FILE_PATH       = DATA_DIR / "laser488nm_5mW_exp500us_rec_HamaCAM.hdf5"
AOD_PATH        = DATA_DIR / "amplituds_fases.mat"
DATASET_ROUTE   = SCRIPT_DIR.parent / "data" / "real" / "processed" / "training_dataset.h5"

# Creates folder structure in case it doesn't exist 
DATASET_ROUTE.parent.mkdir(parents=True, exist_ok=True)

mat_file = AOD_PATH

#################################################33
# LOADING MATLAB AOD DATA

try: # tries to load .mat v7.2 or lower
    datos_mat   = scipy.io.loadmat(mat_file)
    amplitudes  = datos_mat['Asample_stack']
    phases      = datos_mat['Phi_stack']
    print("Read .mat file with scipy.io")

except NotImplementedError: # .mat v7.3 which is as HDF5
    with h5py.File(mat_file, 'r') as f_mat:
        amplitudes  = f_mat['Asample_stack']
        phases      = f_mat['Phi_stack']
        if amplitudes.shape == (38, 4000):
            amplitudes = amplitudes.T
            phases     = phases.T
    print("Read .mat file with h5py")

print(f"Amplitude dimensions: {amplitudes.shape}")
print(f"Phase dimensions: {phases.shape}")

# Known dimensions

NUM_SAMPLES     = 4000
NUM_FREQUENCIES = 38
IMG_HEIGHT      = 40
IMG_WIDTH       = 1844


x_approx_coord  = [178, 209, 240, 271, 302, 333, 365, 396, 427, 458,
                   489, 520, 551, 583, 614, 645, 676, 707, 738, 769,
                   800, 831, 862, 893, 924, 955, 985, 1016, 1047, 1078,
                   1109, 1139, 1170, 1201, 1232, 1262, 1293, 1323]



with h5py.File(FILE_PATH, "r") as f_cam, h5py.File(DATASET_ROUTE, "w") as f_train:

    print(f"Saving dataset in: {DATASET_ROUTE}")
    # Creates empty datasets for all data
    cam_dataset = f_cam['HamaCAM']
    
    X_aod_dataset = f_train.create_dataset("X_aod",
        shape  = (NUM_SAMPLES,NUM_FREQUENCIES, 2), # 2: amplitude and phase. SEE IF FREQUENCIES ARE CORRELATED WITH PEAKS
        dtype  = 'float32',
        chunks = True # use chunks to accelerate batch on nn
    )

    Y_img_dataset = f_train.create_dataset("Y_img",
        shape  = (NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH),
        dtype  = 'int16',
        chunks = True # use chunks to accelerate batch on nn
    )

    Y_int_dataset = f_train.create_dataset("Y_int",
        shape  = (NUM_SAMPLES,NUM_FREQUENCIES),
        dtype  = 'float32',
        chunks = True # use chunks to accelerate batch on nn
    )


    print("Consolidating data...")

    # Filling
    for i in range(NUM_SAMPLES):

        #AOD
        amp_i = amplitudes[i]
        phase_i = phases[i]
        amplitude_pashe_i = np.column_stack((amp_i, phase_i)) # IF NOT CORRELATED np.flip(amplitude_pashe_i, axis=0)

        #IMG
        img_i = cam_dataset[i,:,:]


        intensities_i = integrar_picos_robusto(img_i, x_approx_coord) # FIGURE OUT PEAK_COORDINATE

        X_aod_dataset[i] = amplitude_pashe_i
        Y_img_dataset[i] = img_i
        Y_int_dataset[i] = intensities_i

        if (i + 1) % 500 == 0:
            print(f"Processed {i +1} / {NUM_SAMPLES} samples.")
print("Dataset created successfully.")