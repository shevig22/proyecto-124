# Para capturar el fotograma
import cv2

# Para procesar el arreglo de la imagen
import numpy as np

# Para cargar el modelo preentrenado
import tensorflow as tf

# Adjuntando el índice de la cámara como 0 con la aplicación del software
camera = cv2.VideoCapture(0)

# Cargando el modelo preentrenado: keras_model.h5
mymodel = tf.keras.models.load_model('keras_model.h5')

# Bucle infinito
while True:

	# Leyendo/Solicitando un fotograma de la cámara
	status , frame = camera.read()

	# Si somos capaces de leer exitosamente el fotograma
	if status:

		# Voltear la imagen
		frame = cv2.flip(frame , 1)

		# Redimensionar el fotograma
		resized_frame = cv2.resize(frame , (224,224))

		# Expandir las dimensiones del arreglo a lo largo del eje 0
		resized_frame = np.expand_dims(resized_frame , axis = 0)

		# Normalizar para facilitar el proceso
		resized_frame = resized_frame / 255

		# Obteniendo predicciones del modelo
		predictions = mymodel.predict(resized_frame)

		# Conviritiendo los datos del arreglo en porcentajes de confianza
		rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)

		# Imprimiendo los porcentajes de confianza
		print(f"Piedra: {rock}%, Papel: {paper}%, Tijeras: {scissor}%")

		# Mostrando los fotogramas capturados
		cv2.imshow('Alimentar' , frame)

		# Esperando 1ms
		code = cv2.waitKey(1)
		
		# Si se preciona la barra espaciadora, romper el bucle
		if code == 32:
			break

# Liberar la cámara de la app del software
camera.release()

# Cerrar la ventana abierta
cv2.destroyAllWindows()
