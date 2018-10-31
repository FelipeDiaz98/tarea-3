import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq
from matplotlib.colors import LogNorm

#parte a

datos = plt.imread("arbol.png")

#parte b 

plt.figure()
F = np.fft.fft2(datos)
Fs = np.fft.fftshift(F)
plt.imshow(abs(Fs), cmap = "Blues")
plt.savefig("DiazFelipe_FT2D.pdf")

def filtro(F):
	for i in range(len(F)):
		for j in range(len(F)):
			if ((F[i,j] > 1625) and (F[i,j] < 4130)):
				F[i,j] = 0
	return F

Ff = filtro(F)
Ffs = np.fft.fftshift(Ff)
plt.figure()
plt.imshow(abs(Ffs), cmap = "Blues", norm = LogNorm())
plt.show()

imagen_0 = np.fft.ifft2(Ff)
imagen=np.array(imagen_0)
plt.imsave("filtrada.png", np.real(imagen), cmap="gray")









