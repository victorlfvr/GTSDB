import tensorflow as tf
import matplotlib.pyplot as plt
import json

def entrainer_cnn(model, train_dataset, val_dataset, epochs=30):
    """
    Entraîne le modèle CNN sur les données d'entraînement et validation.
    
    Args:
        model: modèle Keras compilé
        train_dataset: tf.data.Dataset d'entraînement
        val_dataset: tf.data.Dataset de validation
        epochs: nombre d'époques
        
    Returns:
        (model entraîné, historique)
    """
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1
    )
    return model, history


def tracer_courbes_cnn(history, title="CNN - Courbes d'entraînement"):
    """
    Trace les courbes de loss et accuracy pour entraînement et validation.
    
    Args:
        history: objet History de Keras
        title: titre du graphique
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./partie_3/models/cnn_curves.png', dpi=150, bbox_inches='tight')
    print("\nGraphiques sauvegardés: ./partie_3/models/cnn_curves.png")
    plt.show()


def save_model_cnn(model, model_path="./partie_3/models/CNN_64x64.keras"):
    """
    Sauvegarde le modèle CNN.
    
    Args:
        model: modèle Keras
        model_path: chemin de sauvegarde
    """
    model.save(model_path)
    print(f"Modèle sauvegardé: {model_path}")


def save_history_cnn(history, history_path="./partie_3/models/history_cnn_64x64.json"):
    """
    Sauvegarde l'historique d'entraînement en JSON.
    
    Args:
        history: objet History
        history_path: chemin de sauvegarde
    """
    hist_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }
    
    with open(history_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"Historique sauvegardé: {history_path}")


def load_model_cnn(model_path="./partie_3/models/CNN_64x64.keras"):
    """
    Charge un modèle CNN entraîné.
    
    Args:
        model_path: chemin du modèle
        
    Returns:
        modèle Keras ou None si pas trouvé
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Modèle chargé: {model_path}")
        return model
    except Exception as e:
        print(f"erreur: pas de modele - {e}")
        return None
