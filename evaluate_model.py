import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# Paths
# ==========================
DATA_DIR = "data/processed"
MODEL_PATH = "models/basic_cnn_signature_model.h5"  # change if needed

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully")

# ==========================
# Load Test Data
# ==========================
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ==========================
# Evaluate Model
# ==========================
loss, accuracy = model.evaluate(test_data)
print("\n📊 Test Accuracy:", accuracy)
print("📊 Test Loss:", loss)

# ==========================
# Predictions
# ==========================
predictions = model.predict(test_data)
pred_classes = (predictions > 0.5).astype("int32")

true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# ==========================
# Confusion Matrix
# ==========================
cm = confusion_matrix(true_classes, pred_classes)

print("\n📌 Confusion Matrix:")
print(cm)

# ==========================
# Classification Report
# ==========================
print("\n📌 Classification Report:\n")
print(classification_report(true_classes, pred_classes, target_names=class_labels))

# ==========================
# Plot Confusion Matrix
# ==========================
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()
