import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = "models/final_signature_model.keras"
TEST_DIR = "data/processed/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ==========================================
# LOAD MODEL
# ==========================================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully")

# ==========================================
# LOAD TEST DATA
# ==========================================
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ==========================================
# PREDICTIONS
# ==========================================
predictions = model.predict(test_data)
y_pred = (predictions > 0.6).astype(int).ravel()  # Using threshold 0.6
y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Final Model")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add numbers inside matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

# Save image
plt.savefig("confusion_matrix_final.png")
print("\n✅ Confusion matrix saved as confusion_matrix_final.png")

plt.show()
