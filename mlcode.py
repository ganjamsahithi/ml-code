import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# Directory containing the image folders (each folder is a class)
image_dir = '/content/drive/MyDrive/Plant_diseases'  # Update with the correct path

# Load training and validation datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_dir,
    image_size=(224, 224),
    batch_size=32,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='training',
    shuffle=True,
    seed=42
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_dir,
    image_size=(224, 224),
    batch_size=32,
    labels='inferred',
    label_mode='int',
    validation_split=0.4,
    subset='validation',
    shuffle=True,
    seed=42
)

class_names = train_dataset.class_names
num_classes = len(train_dataset.class_names)
print(f'Number of classes: {num_classes}')

# Normalize the pixel values
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Load the InceptionV3 model with pre-trained ImageNet weights
base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Define the model architecture
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=12
)

# Evaluate the model
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation accuracy using InceptionV3: {accuracy * 100:.2f}%")

# Get predictions and true labels
y_true = []
y_pred = []

for images, labels in val_dataset:
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Convert to numpy arrays for metrics calculation
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Print classification report
class_report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", class_report)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()
