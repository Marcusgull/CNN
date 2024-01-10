import tensorflow as tf
from keras import datasets, layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#Last in CIFAR-10 datasettet, dette er nesten likt som MNIST
# men inneholder farger og 10 ulike klasser
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#display noen samples av bildene som CNN modellen skal trenes på
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[i])
    plt.axis('off')
plt.show()

#Pre-prosesser data
train_images, test_images = train_images / 255.0, test_images /255.0

#Endrer på bildene (data-augmentation)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
)


#CNN modellen
#Legger til flere og flere filtre i hver Conv2D layer for å plukke opp mer -
#kompleksivitet i mønstrene.
#MaxPool2D downsampler dimensjonene til outputen, og minsker sjansen for overfitting

#Flatten endrer 3D mappingen til en 1D vektor. Dette klargjør inputen til Dense layer.
#Dense layer er et "fully connected" layer, som lærer globale mønstre og avhengigheter 
# i funksjonsrepresentasjonene som trekkes ut av konvolusjonslagene (liten munnfull)
# Først brukes relu (ikke linjær) for å fange mønstre, så softmax som endrer outputen til sannsynlighets "score"
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Flatten outputten, og legg til Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#Compiler modellen
model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


#Tren modellen
#Ikke bland train og test data sammen, da blir outputten meningsløs
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

#Gridden som man søker gjennom kombinasjoner med hyperparametere
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
epochs_search = [10, 20, 30]

#Grid-search av hyper-parametere
for lr in learning_rates:
    for batch_size_search in batch_sizes:
        for epoch in epochs_search:
            #bygg modellen og tren med hver enkelt kombinasjon
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10, activation='softmax'))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size_search),
                                epochs=epochs_search,
                                validation_data=(test_images, test_labels))

            #evaluer modellen
            test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
            
            #print og lagre resultater
            print(f'Learning Rate: {lr}, Batch Size: {batch_sizes}, Epochs: {epoch}, Test Accuracy: {test_accuracy}')



#Evaluer
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

#Visualiser resultater
predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i])
    predicted_label = tf.argmax(predictions[i]).numpy() #Her jeg kan legge til CUDA? må sjekkes videre
    true_label = test_labels[i, 0]
    plt.title(f'Predicted: {predicted_label}, True: {true_label}')
    plt.axis('off')
plt.show()
#Når siste figuren vises, så vil predicted_label og true_label vise samme eller ulik verdi
#dette vil gjenspeile den kvalitative ytelsen til modellen. Hvis predicted = true
#så betyr det at modellen har "gjettet" korrekt for det tilfellet. Hvis de er ulike i verdi
# betyr det at modellen har miss-klasifisert. I mitt tilfelle fikk jeg resultatet:
# Test accuracy: 0.7175999879837036
# Neste test burde kanskje inneholder bedre bruk av optimizer og generell fine-tuning