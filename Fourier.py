import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftfreq
from scipy import interpolate

#parte a

incompletos = np.genfromtxt("incompletos.dat", delimiter = ",")
signal = np.genfromtxt("signal.dat", delimiter = ",")

#parte b

plt.figure()
plt.plot(signal[:,0], signal[:,1], label = "Senial")
plt.legend(loc = "best")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Grafica de los datos de la senial")
plt.savefig("DiazFelipe_signal.pdf")

#parte c

def fourier(f):
	F = []
	n = len(f)
	for i in range(n):
		g = []
		for k in range(n):
			r = f[k] * np.exp(-1j*2*np.pi*k*i/n)
			g.append(r)
		R = np.sum(g)
		F.append(R)
	return F

F = fourier(signal[:,1])

#parte d

F0 = abs(np.real(F))
n = len(signal[:,0])
dt = ((signal[:,0])[-1] - (signal[:,0])[0])/n
freq = fftfreq(n,dt)

plt.figure()
plt.plot(freq, F0, label = "Transformada")
plt.legend(loc = "best")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Grafica de la transformada de Fourier de la senial")
plt.savefig("DiazFelipe_TF.pdf")

#parte e

print "Las frecuencias principales se encuentran aproximadamente en 0, 139.667, 210.042 y 385.255 Hz, cabe aclarar que cada uno de los picos tiene un 'reflejo' en la parte negativa del eje x con los mismos valores, pero estos son los mismos picos mencionados anteriormente."

#parte f 

def filtro(F,freq,c):
	for i in range (len(freq)):
		if (abs(freq[i]) > c):
			F[i] = 0
	return F

F = filtro(F,freq,1000)
f = np.real(np.fft.ifft(F))
plt.figure()
plt.plot(signal[:,0], f, label = "senial filtrada")
plt.legend(loc = "best")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Grafica senial filtrada")
plt.savefig("DiazFelipe_filtrada.pdf")

#parte g

print "TERMINAR."

#parte h

def interpolacion(datos,array):
	x = datos[:,0]
	y = datos[:,1]
	cuadratica = interpolate.interp1d(x,y,kind = "quadratic")
	cubica = interpolate.interp1d(x,y,kind = "cubic")
	funfcuadratica = cuadratica(array)
	funfcubica = cubica(array)
	return funfcuadratica, funfcubica	

array = np.linspace((incompletos[:,0])[0], (incompletos[:,0])[-1], 512)
cuad, cub = interpolacion(incompletos, array)

cuad = fourier(cuad)
cub = fourier(cub)

#parte i

cuad0 = abs(np.real(cuad))
cub0 = abs(np.real(cub))

n2 = len(array)
dt2 = (array[-1] - array[0])/n2
freq2 = fftfreq(n2,dt2)

plt.figure()
plt.subplots_adjust(hspace=0.8)
plt.subplot(311)
plt.plot(freq, F0, label = "senial original")
plt.legend(loc = "best")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Transformada de senial original")
plt.subplot(312)
plt.plot(freq2, cuad0, label = "interpol cuadratica")
plt.legend(loc = "best")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Transformada de interpolacion cuadratica")
plt.subplot(313)
plt.plot(freq2, cub0, label = "interpol cubica")
plt.legend(loc = "best")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.title("Transformada de interpolacion cubica")
plt.savefig("DiazFelipe_TF_interpola.pdf")

#parte j

print "Se ve en la grafica de las dos interpolaciones que ambas eliminan parte de la senial, eliminando asi el pico que tiene mayor frecuencia (de los picos que antes se describieron como principales), ademas de eso, ambas interpolaciones eliminan la mayor parte del ruido que tenia la senial original, aunque la cuadratica elimina mejor las frecuencias altas y la cuadratica elimina mejor un pequenio range de frecuencias mas bajas, por lo que posiblemente el ajuste cuadratico es mejor para la senial, y el pico que es eliminado era ruido tambien pero de mucha menor frecuencia por lo que el filtro no lo detectaba sin quitar informacion de la senial."

#parte k

F500 = np.real(np.fft.ifft(filtro(F, freq, 500)))
cuad500 = np.real(np.fft.ifft(filtro(cuad, freq2, 500)))
cub500 = np.real(np.fft.ifft(filtro(cub, freq2, 500)))

F1000 = np.real(np.fft.ifft(filtro(F, freq, 1000)))
cuad1000 = np.real(np.fft.ifft(filtro(cuad, freq2, 1000)))
cub1000 = np.real(np.fft.ifft(filtro(cub, freq2, 1000)))

#parte l

plt.figure()
plt.subplots_adjust(hspace=0.5)
plt.subplot(211)
plt.plot(array, F500, label = "senial original")
plt.plot(array, cuad500, label = "interpol cuadratica")
plt.plot(array, cub500, label = "interpol cubica")
plt.legend(loc = "best")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Senial filtrada con pasabajos de 500 Hz")
plt.subplot(212)
plt.plot(array, F1000, label = "senial original")
plt.plot(array, cuad1000, label = "interpol cuadratica")
plt.plot(array, cub1000, label = "interpol cubica")
plt.legend(loc = "best")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title("Senial filtrada con pasabajos de 1000 Hz")
plt.savefig("DiazFelipe_2Filtros.pdf")

