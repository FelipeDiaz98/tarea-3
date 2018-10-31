import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq
from matplotlib.colors import LogNorm
import math

#parte a

datos = plt.imread("arbol.png")

#parte b 

plt.figure()
F = np.fft.fft2(datos)
Fs = np.fft.fftshift(F)
plt.imshow(math.log10(abs(Fs)))
plt.savefig("DiazFelipe_FT2D.pdf")

#parte c

def filtro(F):
	for i in range(len(F)):
		for j in range(len(F)):
			if ((F[i,j] > 1625) and (F[i,j] < 4130)):
				F[i,j] = 0
	return F

Ff = filtro(F)

#parte d

Ffs = np.fft.fftshift(Ff)
plt.figure()
plt.imshow(abs(Ffs), norm = LogNorm())
plt.savefig("DiazFelipe_FT2D_filtrada.pdf")

#parte e

imagen = np.fft.ifft2(Ff)
plt.imsave("DiazFelipe_Imagen_filtrada.pdf", np.real(imagen), cmap="gray")

