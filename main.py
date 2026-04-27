import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import os

# ==========================================
# 1. SETUP AND DATA LOADING
# ==========================================
# Define image dimensions, batch size, and directory paths.
# Load the dataset directly from directories into Train, Validation, and Test splits.
# Apply caching and prefetching (AUTOTUNE) to prevent I/O bottlenecks and maximize GPU efficiency.

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

BASE_DIR = './dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

print("Loading datasets...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=123
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    seed=123
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"\nClasses detected ({num_classes}): {class_names}")

# Count images
num_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
num_val_batches = tf.data.experimental.cardinality(validation_dataset).numpy()
num_test_batches = tf.data.experimental.cardinality(test_dataset).numpy()

print(f"\nDataset size:")
print(f"  Training: ~{num_train_batches * BATCH_SIZE} images")
print(f"  Validation: ~{num_val_batches * BATCH_SIZE} images")
print(f"  Test: ~{num_test_batches * BATCH_SIZE} images")

# Optimization
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# ==========================================
# 2. CALCULATE CLASS WEIGHTS
# ==========================================
# Dynamically analyze the training distribution to detect class imbalance.
# Compute weights to penalize the model heavily for misclassifying minority classes (e.g., elephants)
# and lightly for majority classes, ensuring balanced learning across all categories.

print("\nCalculating class weights for imbalanced dataset...")
all_labels = []
for images, labels in train_dataset.unbatch().batch(BATCH_SIZE * 10):
    all_labels.extend(labels.numpy())
    if len(all_labels) > 5000:  # Muestra suficiente
        break

all_labels = np.array(all_labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

class_weight_dict = dict(enumerate(class_weights))
print("Class weights calculated:")
for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
    print(f"  {class_name}: {weight:.2f}")


# ==========================================
# 3. CUSTOM CNN ARCHITECTURE
# ==========================================
# Build a Custom Convolutional Neural Network from scratch.
# - Includes Data Augmentation layers directly within the model to improve generalization.
# - Uses 4 Convolutional blocks for hierarchical feature extraction.
# - Applies Batch Normalization for training stability and Dropout regularization to prevent overfitting.
# - Ends with Global Average Pooling instead of Flattening for computational efficiency.

print("\nBuilding CNN model...")

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # DATA AUGMENTATION
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
    
    layers.Rescaling(1./255),
    
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Block 4
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Block 5
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Clasiffy
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()


# ==========================================
# 4. MODEL COMPILATION
# ==========================================
# Compile the network using the Adam optimizer with a standard learning rate.
# Use Sparse Categorical Crossentropy as the loss function, which is optimal for mutually exclusive multi-class integer labels.

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


# ==========================================
# 5. TRAINING CALLBACKS
# ==========================================
# Implement dynamic callbacks for optimal training control:
# - EarlyStopping: Halts training if validation loss plateaus, preventing overfitting.
# - ReduceLROnPlateau: Decays the learning rate for fine-grained weight updates when stuck.
# - ModelCheckpoint: Automatically saves only the model state with the highest validation accuracy.

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.3,
    patience=7,
    min_lr=1e-8,
    mode='max',
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)


# ==========================================
# 6. MODEL TRAINING
# ==========================================
# Execute the training loop for up to 80 epochs.
# The class_weight parameter is injected here to force the model to pay equal attention to underrepresented classes.

EPOCHS = 80
print(f"\nStarting training for up to {EPOCHS} epochs...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)


# ==========================================
# 7. FINAL EVALUATION
# ==========================================
# Evaluate the fully trained model against the completely unseen test dataset to assess real-world generalization capability.

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"\n✓ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✓ Test Loss: {test_loss:.4f}")


# ==========================================
# 8. CONFUSION MATRIX VISUALIZATION
# ==========================================
# Generate predictions for the entire test set.
# Compute and plot a visual heatmap of the Confusion Matrix to deeply analyze class-specific misclassifications and model biases.

print("\nGenerating confusion matrix...")
y_pred = []
y_true = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Animal Classification', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")


# ==========================================
# 9. INFERENCE EXAMPLES
# ==========================================
# Extract a batch of unseen test images to perform raw predictions.
# Generate a 3x4 visual grid displaying the image, predicted class, confidence score, and true label (Green = Correct, Red = Incorrect).

print("Generating prediction examples...")
plt.figure(figsize=(16, 12))

sample_count = 0
for images, labels in test_dataset:
    if sample_count >= 12:
        break
    
    predictions = model.predict(images, verbose=0)
    batch_size = min(12 - sample_count, len(images))
    
    for i in range(batch_size):
        plt.subplot(3, 4, sample_count + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_idx = np.argmax(predictions[i])
        predicted_class = class_names[predicted_idx]
        true_class = class_names[labels[i]]
        confidence = predictions[i][predicted_idx] * 100
        
        color = 'green' if predicted_class == true_class else 'red'
        plt.title(f"Pred: {predicted_class} ({confidence:.1f}%)\nTrue: {true_class}", 
                 color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
        sample_count += 1

plt.suptitle('Prediction Examples (Green=Correct, Red=Wrong)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prediction_examples.png")


# ==========================================
# 10. TRAINING PROGRESS GRAPHS
# ==========================================
# Extract metrics history from the training loop.
# Plot Training vs Validation Accuracy and Loss over time to visually diagnose the learning curve and model convergence.

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(16, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-', label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
random_baseline = 100.0 / num_classes
plt.axhline(y=random_baseline/100, color='gray', linestyle='--', 
            alpha=0.5, label=f'Random Guess ({random_baseline:.1f}%)')
plt.legend(loc='lower right', fontsize=11)
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
plt.legend(loc='upper right', fontsize=11)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_graphs.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_graphs.png")


# ==========================================
# 11. DETAILED TEXT REPORT GENERATION
# ==========================================
# Automatically compile and export a comprehensive text file (training_report.txt).
# Logs all experiment hyperparameters, dataset splits, best accuracies, and Scikit-Learn's detailed per-class metrics.

print("\nGenerating detailed report...")
with open('training_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write(" ANIMAL CLASSIFICATION - TRAINING REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Classes ({num_classes}): {', '.join(class_names)}\n")
    f.write(f"Image size: {IMG_HEIGHT}x{IMG_WIDTH} pixels\n")
    f.write(f"Batch size: {BATCH_SIZE}\n\n")
    
    f.write(f"Training images: ~{num_train_batches * BATCH_SIZE}\n")
    f.write(f"Validation images: ~{num_val_batches * BATCH_SIZE}\n")
    f.write(f"Test images: ~{num_test_batches * BATCH_SIZE}\n\n")
    
    f.write(f"Epochs trained: {len(acc)}/{EPOCHS}\n")
    f.write(f"Best training accuracy: {max(acc) * 100:.2f}%\n")
    f.write(f"Best validation accuracy: {max(val_acc) * 100:.2f}%\n")
    f.write(f"Final test accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Final test loss: {test_loss:.4f}\n\n")
    
    f.write("Class Weights (for imbalanced dataset):\n")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        f.write(f"  {name}: {weight:.3f}\n")
    f.write("\n")
    
    f.write("=" * 70 + "\n")
    f.write(" CLASSIFICATION REPORT (Per-Class Metrics)\n")
    f.write("=" * 70 + "\n\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    f.write("\n" + "=" * 70 + "\n")
    f.write(" CONFUSION MATRIX\n")
    f.write("=" * 70 + "\n\n")
    f.write("         " + "  ".join([f"{name:>8}" for name in class_names]) + "\n")
    for i, row in enumerate(cm):
        f.write(f"{class_names[i]:>8} " + "  ".join([f"{val:>8}" for val in row]) + "\n")

print("✓ Saved: training_report.txt")


print("\n" + "=" * 70)
print(" TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\n Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f" Random Baseline: {100.0/num_classes:.2f}%")
print(f" Improvement: {(test_accuracy * 100) - (100.0/num_classes):.2f} percentage points")
print("\n Generated files:")
print("training_graphs.png")
print("confusion_matrix.png")
print("prediction_examples.png")
print("training_report.txt")
print("best_model.keras")
print("\n" + "=" * 70 + "\n")