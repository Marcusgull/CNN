import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt

#Last inn MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#Viser 25 eksempler fra MNIST datasettet i et 5x5 rutenett i grå-skala
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.axis('off')
plt.show()

#Preprosseserer data

#Normaliser pixle verdi til mellom 0 og 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#Endre bildet til (28, 28, 1) siden CNN forventer 3D input
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#bygg CNN modellen med convolutional layers og max-pooling layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#compiler modellen
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',                         #adam optimizer er veldig enkelt og grei å bruke :D
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#tren modellens data med 10 epochs, den bruker validerings data for evaluering. Epochs kan justeres senere
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

#Evaluer modellen, og print test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

#Gjett hva du får!
predictions = model.predict(test_images)

#Visualiser resultatet
import numpy as np

# Funksjon som viser det antat svar, og den ekte merkelappen med bildet
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.reshape((28, 28)), cmap='gray')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f'{predicted_label} ({true_label})', color=color)


# Plotter de første 15 test bildene med antatte merkelapper, med den ekte merkelappen
num_rows, num_cols = 3, 5
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
plt.show()

