# Importing libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Check the shape of the dataset
print(f"Training data shape: {train_images.shape}, Test data shape: {test_images.shape}")

# Defining CNN Model
model = models.Sequential([
    # Convolutional Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Convolutional Layer 3
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten the 3D outputs to 1D
    layers.Flatten(),
    
    # Fully connected layer
    layers.Dense(64, activation='relu'),
    
    # Output layer (10 classes)
    layers.Dense(10)
])

# Print the model summary
model.summary()

# Compiling
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Performance on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing Accuracy')

# Plot training & validation loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Testing Loss')

plt.show()

# Predict on the first 5 test images
predictions = model.predict(test_images[:5])

# Display the images and their predicted labels
for i in range(5):
    plt.imshow(test_images[i])
    plt.title(f"Predicted label: {np.argmax(predictions[i])}")
    plt.show()

model.save('cifar10_cnn_model.h5')
