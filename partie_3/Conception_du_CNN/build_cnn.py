import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(num_classes, input_shape=(64, 64, 3)):
    """
    Construit un modèle CNN pour la classification de panneaux routiers.
    
    Architecture:
    - Conv2D (32 filtres) + ReLU + MaxPooling
    - Conv2D (64 filtres) + ReLU + MaxPooling
    - Conv2D (128 filtres) + ReLU
    - Flatten
    - Dense (128) + ReLU + Dropout(0.5)
    - Dense (num_classes) + Softmax
    
    Args:
        num_classes: nombre de classes (sortie softmax)
        input_shape: shape des images (H, W, C), par défaut (64, 64, 3)
        
    Returns:
        modèle compilé
    """
    model = models.Sequential([
        # Bloc 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 3 (optionnel, améliore extraction de features)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        
        # Projection et couches denses
        layers.Flatten(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile le modèle avec une fonction de coût et un optimiseur.
    
    Args:
        model: modèle Keras
        learning_rate: taux d'apprentissage Adam
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
