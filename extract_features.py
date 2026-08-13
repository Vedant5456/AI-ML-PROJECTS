from tensorflow.keras.models import load_model, Model

base_model = load_model("models/final_signature_model.keras")

feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output
)

def get_features(img):

    features = feature_model.predict(img)

    return features.flatten()