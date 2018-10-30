import numpy as np
import matplotlib.pyplot as plt

#datos = wget.download("http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat")

#parte a
datos = np.genfromtxt("datos.dat", delimiter = ",", usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))
diagnostico = np.genfromtxt("datos.dat", delimiter = ",", usecols = 1, dtype = "string")

#parte b
arr = []
for i in range (len(datos[0])):
	A=[]
	for j in range (len(datos[0])):
		resultados = []
		sumai = np.average(datos[:,i])
		sumaj = np.average(datos[:,j])
		for k in range (len(datos)):
			valor = (((datos[:,i])[k] - sumai)*((datos[:,j])[k] - sumaj))/(len(datos)-1)
			resultados.append(valor)
		A.append(np.sum(resultados))
	arr.append(A)

arr = np.array(arr)
print arr

#parte c
valores, vectores = np.linalg.eig(arr)

for i in range(3):
	for j in range (10):

		string = ""

		if (i == 0):
			string = string + "mean "
		if (i == 1):
			string = string + "standard error - "
		if (i == 2):
			string = string + "worst "
		if (j == 0):
			string = string + "radius:"
		if (j == 1):
			string = string + "texture:"
		if (j == 2):
			string = string + "perimeter:"
		if (j == 3):
			string = string + "area:"
		if (j == 4):
			string = string + "smoothness:"
		if (j == 5):
			string = string + "compactness:"
		if (j == 6):
			string = string + "concavity:"
		if (j == 7):
			string = string + "concave points:"
		if (j == 8):
			string = string + "symmetry:"
		if (j == 9):
			string = string + "fractal dimension:"
		
		print string
		print valores[(i*10)+j]
		print vectores[(i*10)+j]

#parte d
print "Las dos componentes mas importantes son mean radius y mean texture que corresponde a las primeras dos columnas de los datos."

#parte e
PC1 = np.dot(datos, vectores[0])
PC2 = np.dot(datos, vectores[1])

mx = []
my = []
bx = []
by = []
for i in range(len(diagnostico)):	
	if (diagnostico[i] == "M"):
		mx.append(PC1[i])
		my.append(PC2[i])
	if (diagnostico[i] == "B"):
		bx.append(PC1[i])
		by.append(PC2[i])

plt.figure()
plt.scatter(mx,my,c="red",label="Maligno")
plt.scatter(bx,by,c="blue",label="Benigno")
plt.legend(loc="best")
plt.xlabel("PC1 (mean radius)")
plt.ylabel("PC2 (mean texture)")
plt.title("Diagnostico en funcion de las dos componentes principales")
plt.savefig("DiazFelipe_PCA.pdf")

#parte f
print "Creo que el metodo si sirve para poder ayudar al diagnostico de si el paciente tiene un tumor benigno o maligno, cuando se mira la grafica de las dos componentes principales, pareciera que la mayoria de los datos van en diagonal mostrando una relacion inversamente proporcional entre las dos variables, se podria sacar una linea de ajuste lineal entre las dos variables y ver en que parte de la linea esta el paciente: la superior izquiera significa que el paciente no tiene cancer, y la inferior derecha significa que el paciente tiene cancer, aunque cabe aclarar que dicho ajuste seria mas preciso entre mas a los extremos este el paciente, donde pareciera que el metodo tiene menos fallas (diagnosticos que no sean acordes a esta aproximacion), ya que en la parte donde se unen ambas mitades (los resultados benignos y malignos) no estaria muy claro si la persona tiene o no cancer ya que hay algunos datos en dicho punto que no siguen esta aproximacion de manera tan exacta. Para mejorar el metodo se podrian tomar muchos mas datos con el fin de hacer mas claro lo que pasa en el punto donde se unen ambos diagnosticos, y ver si se podria aumentar la exactitud del metodo al reducir esta franja de incertidumbre."












