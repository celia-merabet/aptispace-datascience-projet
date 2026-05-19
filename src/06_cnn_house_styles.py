"""
06_cnn_house_styles.py — CNN de Classification des Styles de Maisons

Objectif : Classifier les maisons en 3 catégories de prix (Économique, Moyenne, Luxe)
à partir de leurs photos, en utilisant un réseau de neurones convolutif (CNN) sous TensorFlow.

Conforme à l'énoncé du Jalon 2 — Chapitre 5 :
"Intégration d'une brique de Deep Learning (traitement d'images) à l'aide de
réseaux de neurones convolutifs (CNN) sous TensorFlow pour classifier vos prédictions."

NOTE : L'entraînement est réalisé sur Google Colab (faute de TensorFlow installable
sur Windows local). Ce script reproduit l'architecture et le pipeline complet.
Le dataset utilisé : House Prices and Images (SoCal) — 15 474 maisons californiennes.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("🧠 CNN Classification — Démarrage")

# =====================================================
# 1. CONFIGURATION
# =====================================================
N_IMAGES = 1000      # Nombre d'images utilisées (limité pour démo)
IMG_SIZE = 128       # Dimensions des images
EPOCHS = 10
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "socal2.csv")
IMG_DIR = os.path.join(BASE_DIR, "data", "raw", "socal_pics")

# =====================================================
# 2. CHARGEMENT DES IMAGES ET DES PRIX
# =====================================================
df = pd.read_csv(DATA_PATH)
df_sample = df.head(N_IMAGES).copy()

X_images = []
y_prices = []

for idx, row in df_sample.iterrows():
    img_path = os.path.join(IMG_DIR, f"{row['image_id']}.jpg")
    if os.path.exists(img_path):
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        X_images.append(img_array)
        y_prices.append(row['price'])

X_images = np.array(X_images)
y_prices = np.array(y_prices)

print(f"✔ {len(X_images)} images chargées — Shape : {X_images.shape}")

# =====================================================
# 3. CRÉATION DES CATÉGORIES DE PRIX (3 CLASSES)
# =====================================================
seuil_eco = np.quantile(y_prices, 0.33)
seuil_lux = np.quantile(y_prices, 0.66)

def categoriser_prix(prix):
    if prix < seuil_eco:
        return 0  # Économique
    elif prix < seuil_lux:
        return 1  # Moyenne
    else:
        return 2  # Luxe

y_categories = np.array([categoriser_prix(p) for p in y_prices])

print(f"✔ Seuils : Éco < ${seuil_eco:,.0f} | Moyenne | Luxe > ${seuil_lux:,.0f}")

# =====================================================
# 4. SPLIT TRAIN / TEST
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_categories,
    test_size=0.2,
    random_state=42,
    stratify=y_categories
)

print(f"✔ Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# =====================================================
# 5. ARCHITECTURE DU CNN
# =====================================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes : Éco, Moyenne, Luxe
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✔ Architecture CNN construite")
model.summary()

# =====================================================
# 6. ENTRAÎNEMENT
# =====================================================
print("\n🚀 Entraînement en cours...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# =====================================================
# 7. ÉVALUATION
# =====================================================
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy test : {acc*100:.2f}%")
print("\n📊 Rapport de classification :")
print(classification_report(
    y_test, y_pred,
    target_names=['Économique', 'Moyenne', 'Luxe']
))

print("\n🔢 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# 8. SAUVEGARDE
# =====================================================
output_dir = os.path.join(BASE_DIR, "models")
os.makedirs(output_dir, exist_ok=True)
model.save(os.path.join(output_dir, "cnn_house_classifier.keras"))

predictions_df = pd.DataFrame({
    'image_id': df_sample.iloc[:len(y_categories)]['image_id'].values,
    'predicted_category': np.argmax(model.predict(X_images), axis=1),
    'true_category': y_categories
})
predictions_df.to_csv(
    os.path.join(BASE_DIR, "data", "processed", "cnn_predictions.csv"),
    index=False
)

print("\n💾 Modèle sauvegardé : models/cnn_house_classifier.keras")
print("💾 Prédictions sauvegardées : data/processed/cnn_predictions.csv")
print("\n🚀 CNN Classification terminée avec succès")