import torch
import torch.nn.functional as f


# Metrics for forward model. AOD ---> Image


def psnr(img_pred, img_real, data_range=1.0):
    '''
    Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Typical acceptable values: > 25-30 dB.
    '''
    mse = F.mse_loss(img_pred, img_real)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def peak_energy_error(intensities_pred, intensities_real):
    '''
    Absolute relative average error (%) in the energy sum of 38 peaks,
    Measures if network distributes light correctly on points on interest.
    '''
    # Avoid division by 0
    mask = intensities_real > 1e-5
    mape = torch.mean(torch.abs((intensities_real[mask] - intensities_pred[mask]) / intensities_real[mask])) * 100
    return mape

# Metrics for backward model. Image <--- AOD

def circular_phase_error(phase_pred, phase_real):
    '''
    Measures angular error in rad (range(0,PI)).
    Error = 1 - cos(phase_pred - phase_real)
    '''
    angular_difference = 1-0 - torch.cos(phase_pred - phase_real)
    return torch.mean(angular_difference)

def evaluation_metrics_aod(aod_pred, aod_real):
    '''
    Computes mean absolute error separated for amplitudes and circular error for phases.
    aod_pred and aod_real have dimensions (Batch, 38, 2) [amp, phase]
    '''
    amp_pred, amp_real = aod_pred[:, :, 0], aod_real[:, :, 0]
    phase_pred, phase_real = aod_pred[:, :, 1], aod_real[:, :, 1]

    mae_amplitude = F.l1_loss(amp_pred, amp_real)
    phase_error = circular_phase_error(phase_pred, phase_real)

    return {
        "MAE_Amplitude": mae_amplitude.item(),
        "Angular_Phase_Error": phase_error.item()
    }