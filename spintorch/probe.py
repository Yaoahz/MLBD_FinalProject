import torch
import skimage


class WaveProbe(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		self.register_buffer('x', torch.tensor(x, dtype=torch.int64))
		self.register_buffer('y', torch.tensor(y, dtype=torch.int64))

	def forward(self, m):
		return m[:,0, self.x, self.y]

	def coordinates(self):
		return self.x.cpu().numpy(), self.y.cpu().numpy()

class WaveIntensityProbe(WaveProbe):
	def __init__(self, x, y):
		super().__init__(x, y)

	def forward(self, m):
		return super().forward(m).pow(2)

# default version
# class WaveIntensityProbeDisk(WaveProbe): 
# 	def __init__(self, x, y, r):
# 		x, y = skimage.draw.disk((x, y), r)
# 		super().__init__(x, y)

# 	def forward(self, m):
# 		return super().forward(m).sum().pow(2).unsqueeze(0)

class WaveIntensityProbeDisk(WaveProbe):
	def __init__(self, x, y, r):
		x, y = skimage.draw.disk((x, y), r)
		super().__init__(x, y)

	def forward(self, m):
		return super().forward(m).abs().sum().pow(2).unsqueeze(0)

class WavePhaseProbeDisk(WaveProbe):
    """Wave Probe that captures phase information within a circular region."""
    def __init__(self, x, y, r):
        x, y = skimage.draw.disk((x, y), r)  # Generate circular coordinates
        super().__init__(x, y)

    def forward(self, m):
        """Extract phase information within the disk region."""
        print(f"DEBUG: Input tensor shape: {m.shape}")  # Debugging info

        if m.dim() == 4 and m.shape[1] == 3:  # Expected shape: [batch, channels, x, y]
            max_x, max_y = m.shape[2] - 1, m.shape[3] - 1  # Get grid size
            
            # Clamp indices to prevent out-of-bounds errors
            x = torch.clamp(self.x, 0, max_x)  
            y = torch.clamp(self.y, 0, max_y)  

            # Extract the `mz` component (assuming it is in channel index 2)
            mz_values = m[:, 2, x, y]  # Extract m_z values for phase computation

            # Compute phase using FFT
            analytic_signal = torch.fft.fft(mz_values, dim=0)  
            phase = torch.angle(analytic_signal)  # Extract phase
            
            return phase.mean().unsqueeze(0)  # Return average phase in region
        else:
            raise ValueError(f"Unexpected tensor shape: {m.shape}, check indexing!")


