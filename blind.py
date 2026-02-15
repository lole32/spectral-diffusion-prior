# Author: JianJun Liu
# Date: 2021/12/27
# Modified: 2024 - Added support for fixed proper SRF
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun

from utils.toolkits import toolkits
from utils.torchkits import torchkits
from utils.blur_down import BlurDown
from data.data_info import DataInfo


class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio, fixed_srf=None):
        """
        Blind estimation network for PSF and SRF.
        
        Parameters:
        -----------
        hs_bands : int
            Number of hyperspectral bands
        ms_bands : int
            Number of multispectral bands
        ker_size : int
            Size of PSF kernel
        ratio : int
            Spatial downsampling ratio
        fixed_srf : torch.Tensor or None
            If provided, use this fixed SRF instead of learning it.
            Shape: (ms_bands, hs_bands, 1, 1)
        """
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio = ratio
        self.pad_num = int((self.ker_size - 1) / 2)
        
        # PSF - always learned
        psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))
        self.psf = nn.Parameter(psf)
        
        # SRF - can be fixed or learned
        if fixed_srf is not None:
            # Use provided proper SRF (NOT trainable)
            print("  Using FIXED proper SRF (not trainable)")
            self.fixed_srf = True
            # Ensure correct shape
            if fixed_srf.ndim == 2:
                # Input is (ms_bands, hs_bands), reshape to (ms_bands, hs_bands, 1, 1)
                fixed_srf = fixed_srf.reshape(self.ms_bands, self.hs_bands, 1, 1)
            self.srf = fixed_srf  # NOT a Parameter, just a tensor
        else:
            # Learn SRF from scratch (uniform initialization)
            print("  Learning SRF from scratch (uniform initialization)")
            self.fixed_srf = False
            srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands)
            self.srf = nn.Parameter(srf)
        
        self.blur_down = BlurDown()

    def forward(self, Y, Z):
        """
        Forward pass: predict MS from HS and HS from MS
        
        Parameters:
        -----------
        Y : torch.Tensor
            Hyperspectral image (1, hs_bands, h, w)
        Z : torch.Tensor
            Multispectral image (1, ms_bands, H, W)
        
        Returns:
        --------
        Ylow : torch.Tensor
            Y projected to MS space then normalized
        Zlow : torch.Tensor
            Z blurred and downsampled to HS resolution
        """
        # For fixed SRF, ensure it's on the right device
        if self.fixed_srf:
            srf = self.srf.to(Y.device)
        else:
            srf = self.srf
        
        # Y → MS: Apply spectral response function
        srf_div = torch.sum(srf, dim=1, keepdim=True)
        srf_div = torch.div(1.0, srf_div)
        srf_div = torch.transpose(srf_div, 0, 1)  # 1 x l x 1 x 1
        Ylow = fun.conv2d(Y, srf, None)
        Ylow = torch.mul(Ylow, srf_div)
        Ylow = torch.clamp(Ylow, 0.0, 1.0)
        
        # Z → HS resolution: Blur and downsample
        Zlow = self.blur_down(Z, self.psf, self.pad_num, self.ms_bands, self.ratio)
        Zlow = torch.clamp(Zlow, 0.0, 1.0)
        
        return Ylow, Zlow


class Blind(DataInfo):
    def __init__(self, ndata, nratio, nsnr=0, blind=True, lr=1e-5, use_proper_srf=True):
        """
        Blind estimation of PSF and SRF.
        
        Parameters:
        -----------
        ndata : int
            Dataset index
        nratio : int
            Spatial ratio
        nsnr : int
            Noise level index
        blind : bool
            If False, use ground truth PSF/SRF (for synthetic data)
        lr : float
            Learning rate
        use_proper_srf : bool
            If True and SRF is available in data, use it as fixed (don't learn)
            If False, always learn SRF from scratch
        """
        super().__init__(ndata, nratio, nsnr)
        self.strBR = 'BR.mat'
        self.blind = blind
        self.use_proper_srf = use_proper_srf
        
        if self.blind is False:
            # Use ground truth PSF/SRF (for synthetic benchmarks)
            print('Using true PSF and SRF from ground truth!')
            return
        
        print('Blind estimation of PSF and SRF...')
        
        # Check if proper SRF is provided in data
        fixed_srf_tensor = None
        if self.use_proper_srf and hasattr(self, 'srf') and self.srf is not None:
            # Check if this is a proper SRF (not uniform ones)
            if isinstance(self.srf, np.ndarray):
                # Check if it's non-uniform
                if not np.allclose(self.srf, self.srf.mean()):
                    print('  ✅ Proper SRF detected in data!')
                    print(f'     SRF shape: {self.srf.shape}')
                    print(f'     Will use FIXED SRF (only learn PSF)')
                    
                    # Convert to torch tensor with correct shape
                    if self.srf.ndim == 2:
                        # (ms_bands, hs_bands) → (ms_bands, hs_bands, 1, 1)
                        srf_np = self.srf.reshape(self.ms_bands, self.hs_bands, 1, 1)
                    elif self.srf.ndim == 4:
                        # Already correct shape
                        srf_np = self.srf
                    else:
                        # (hs_bands, ms_bands) → transpose and reshape
                        srf_np = self.srf.T.reshape(self.ms_bands, self.hs_bands, 1, 1)
                    
                    fixed_srf_tensor = torch.tensor(srf_np, dtype=torch.float32)
                else:
                    print('  ⚠️  Uniform SRF detected - will learn from scratch')
            else:
                print('  ⚠️  No proper SRF in data - will learn from scratch')
        else:
            print('  Learning both PSF and SRF from scratch')
        
        # Set parameters
        self.lr = lr
        self.ker_size = 2 * self.ratio - 1
        
        # Store input data
        self.__hsi = torch.tensor(self.hsi)
        self.__msi = torch.tensor(self.msi)
        
        # Create model
        self.model = BlindNet(
            self.hs_bands, 
            self.ms_bands, 
            self.ker_size, 
            self.ratio,
            fixed_srf=fixed_srf_tensor
        ).cuda()
        
        # Optimizer - only optimize PSF if SRF is fixed
        if fixed_srf_tensor is not None:
            # Only optimize PSF
            self.optimizer = optim.Adam([self.model.psf], lr=self.lr)
            print(f'  Optimizing: PSF only ({self.model.psf.numel()} parameters)')
        else:
            # Optimize both PSF and SRF
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f'  Optimizing: PSF + SRF ({total_params} parameters)')
        
        toolkits.check_dir(self.model_save_path)

    def train(self, max_iter=5000, verb=True):
        """
        Train the blind estimation network.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations
        verb : bool
            Verbose output
        """
        if self.blind is False:
            return
        
        hsi, msi = self.__hsi.cuda(), self.__msi.cuda()
        
        print(f"\nStarting blind estimation training ({max_iter} iterations)...")
        
        for epoch in range(0, max_iter):
            Ylow, Zlow = self.model(hsi, msi)
            loss = torchkits.torch_norm(Ylow - Zlow)
            
            if verb is True:
                if (epoch + 1) % 100 == 0:
                    loss_val = loss.item()
                    print(f'  Epoch {epoch + 1:5d}/{max_iter}, lr: {self.lr:.2e}, loss: {loss_val:.6f}')
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.apply(self.check_weight)
        
        # Save model
        torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')
        
        # Extract PSF and SRF
        self.psf = torch.tensor(self.model.psf.data.cpu().detach().numpy())
        
        if self.model.fixed_srf:
            # Use the fixed SRF
            self.srf = self.model.srf.cpu().detach()
            print("  ✅ Used FIXED proper SRF")
        else:
            # Use the learned SRF
            self.srf = torch.tensor(self.model.srf.data.cpu().detach().numpy())
            print("  ⚠️  Used LEARNED SRF (may not be physically accurate)")

    def get_save_result(self, is_save=True):
        """
        Save PSF and SRF to disk.
        
        Parameters:
        -----------
        is_save : bool
            Whether to save to .mat file
        """
        if self.blind is False:
            return
        
        print('\nSaving PSF and SRF...')
        self.model.load_state_dict(torch.load(self.model_save_path + 'parameter.pkl'))
        
        psf = self.model.psf.data.cpu().detach().numpy()
        
        if self.model.fixed_srf:
            srf = self.model.srf.cpu().detach().numpy()
            print('  Using FIXED proper SRF')
        else:
            srf = self.model.srf.data.cpu().detach().numpy()
            print('  Using LEARNED SRF')
        
        psf = np.squeeze(psf)
        srf = np.squeeze(srf)  # (ms_bands, hs_bands)
        
        self.psf, self.srf = psf, srf
        
        # Report SRF characteristics
        print(f'\nSRF characteristics:')
        print(f'  Shape: {srf.shape}')
        print(f'  Range: [{srf.min():.6f}, {srf.max():.6f}]')
        
        for i in range(min(4, srf.shape[0])):
            peak_band = np.argmax(srf[i, :])
            active_bands = np.sum(srf[i, :] > 0.01)
            print(f'  MS band {i}: peak at EnMAP band {peak_band}, '
                  f'{active_bands} active bands (>1%)')
        
        if is_save is True:
            sio.savemat(self.save_path + self.strBR, {'B': psf, 'R': srf})
            print(f'\n✅ Saved to: {self.save_path + self.strBR}')
        
        return

    @staticmethod
    def check_weight(model):
        """
        Ensure PSF and SRF weights remain valid during training.
        """
        # Constrain PSF
        if hasattr(model, 'psf'):
            w = model.psf.data
            w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)
            psf_div = torch.div(1.0, psf_div)
            w.mul_(psf_div)
        
        # Constrain SRF (only if it's trainable)
        if hasattr(model, 'srf') and isinstance(model.srf, nn.Parameter):
            w = model.srf.data
            w.clamp_(0.0, 10.0)
            srf_div = torch.sum(w, dim=1, keepdim=True)
            srf_div = torch.div(1.0, srf_div)
            w.mul_(srf_div)
