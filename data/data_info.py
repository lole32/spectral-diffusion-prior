"""
2023/06/26
"""
import numpy as np
import scipy.io as sio
from utils.toolkits import toolkits

class DataInfo:
    """
        file structure
        ./
        ../pavia/
        ../ksc/
        ../dc/
        .../pavia/XXX/
        .../pavia/Blind/
        .../pavia/pavia_data_r8_20_30.mat
        ..../pavia/Blind/r8_20_30/
    """
    def __init__(self, ndata=0, nratio=8, nsnr=0):
        name = self.__class__.__name__
        print('%s is running' % name)
        # 👇 Point to your local EnMAP+S2 .mat
        self.gen_path = "/home/jovyan/spectral-diffusion-prior/spectral_diffusion_prior/images/"
        self.folder_names = ['enmap/']  # single folder for now
        self.data_names = ['enmap_s2_data_r']  # base name without r3 and .mat
        # keep the original noise list so nsnr=6 -> '' (no noise suffix)
        self.noise = ['_20_30', '_25_35', '_30_40', '_35_45', '_40_50', '_50_60', '']
        # nratio will be 6; nsnr=6 means no extra suffix
        self.file_path = (
            self.gen_path
            + self.folder_names[ndata]
            + self.data_names[ndata]
            + str(nratio)
            + self.noise[nsnr]
            + '.mat'
        )
        print("Loading:", self.file_path)
        mat = sio.loadmat(self.file_path)
        hsi, msi = mat['I_HS'], mat['I_MS']
        
        if 'I_REF' in mat.keys():
            ref = mat['I_REF']  # H x W X L
        else:
            ref = np.ones(shape=(msi.shape[0], msi.shape[1], hsi.shape[2]))
        
        if 'TGT' in mat.keys():
            tgt = mat['TGT']  # H x W X L
        else:
            tgt = hsi
        
        # if 'K' in mat.keys():
        #     psf, srf = mat['K'], mat['R']  # K x K, l X L
        # else:
        #     psf = np.ones(shape=(msi.shape[0] // hsi.shape[0], msi.shape[1] // hsi.shape[1]))
        #     srf = np.ones(shape=(msi.shape[-1], hsi.shape[-1]))

        if 'K' in mat.keys():
            psf = mat['K']
        else:
            # Create dummy PSF (will be estimated anyway)
            psf = np.ones(shape=(msi.shape[0] // hsi.shape[0], 
                                 msi.shape[1] // hsi.shape[1]))
        
        # Check SRF separately!
        if 'R' in mat.keys():
            srf = mat['R']
            print("✅ SRF loaded from file")
        else:
            # Create dummy SRF
            srf = np.ones(shape=(msi.shape[-1], hsi.shape[-1]))
            print("⚠️  No SRF in file, creating uniform")
        
        self.save_path = self.gen_path + self.folder_names[ndata] + name + '/r' + str(nratio) + self.noise[
            nsnr] + '/'
        
        hsi = hsi.astype(np.float32)
        msi = msi.astype(np.float32)
        ref = ref.astype(np.float32)
        tgt = tgt.astype(np.float32)
        self.psf = psf.astype(np.float32)
        self.srf = srf.astype(np.float32)
        self.model_save_path = self.save_path + 'model/'
        
        # preprocess
        self.hsi = toolkits.channel_first(hsi)  # 1 x L x h x w
        self.msi = toolkits.channel_first(msi)  # 1 x L x H x W
        self.tgt = toolkits.channel_first(tgt)  # 1 x l x H x W
        
        # ⚠️ REPLACE THE COMMENTED LINE WITH THIS ⚠️
        # For real data: set self.ref = None (no ground truth exists)
        # For synthetic data: set self.ref from I_REF key
        if 'I_REF' in mat.keys():
            # Check if it's real reference data or just dummy ones
            if np.max(ref) > 1.01 or np.std(ref) > 0.01:
                # Real reference data from synthetic benchmark
                self.ref = toolkits.channel_first(ref)  # 1 x l x H x W
                print("Reference image loaded (synthetic data)")
            else:
                # Dummy reference (all ones) -> no real ground truth
                self.ref = None
                print("No reference image (real data)")
        else:
            # No I_REF key at all
            self.ref = None
            print("No reference image (real data)")
        
        self.hs_bands, self.ms_bands = self.hsi.shape[1], self.msi.shape[1]
        self.ratio = int(self.msi.shape[-1] / self.hsi.shape[-1])
        self.height, self.width = self.msi.shape[2], self.msi.shape[3]
        
        pass