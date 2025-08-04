import numpy as np

# 生成一个信号
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
X = np.fft.fft(x)       # DFT
x_recon = np.fft.ifft(X)  # 逆DFT
print("Original signal:", x)
print("length of signal:", len(x))
print("DFT of signal:", X)
print("length of DFT:", len(X))
print("Reconstructed signal:", x_recon)
print("length of reconstructed signal:", len(x_recon))
