import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from partie_3.Preparation_des_donnees.prepare_data_cnn import decode_class_y


def evaluer_performances_cnn(model, train_ds, test_ds):
    """
    Évalue les performances du modèle CNN sur train et test.
    
    Args:
        model: modèle Keras
        train_ds: tf.data.Dataset d'entraînement
        test_ds: tf.data.Dataset de test
    """
    print("\n" + "="*50)
    print("--- Performances CNN ---")
    print("="*50)
    
    # Évaluation TRAIN
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    print(f"\nPerformances TRAIN:")
    print(f"  Loss:     {train_loss:.4f}")
    print(f"  Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # Évaluation TEST
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nPerformances TEST:")
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\nDifférence Train-Test: {(train_acc - test_acc)*100:.2f}% (overfitting)")
    print("="*50 + "\n")


def get_predictions_cnn(model, dataset):
    """
    Récupère les prédictions et vraies labels sur un dataset.
    
    Args:
        model: modèle Keras
        dataset: tf.data.Dataset
        
    Returns:
        (y_true, y_pred) : arrays numpy
    """
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in dataset:
        y_true.extend(y_batch.numpy())
        predictions = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
    
    return np.array(y_true), np.array(y_pred)


def afficher_matrice_confusion_cnn(y_true, y_pred, title="Matrice de confusion CNN"):
    """
    Affiche et sauvegarde la matrice de confusion.
    
    Args:
        y_true: labels vrais
        y_pred: labels prédits
        title: titre du graphique
        
    Returns:
        matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig('./partie_3/models/confusion_matrix_cnn.png', dpi=150, bbox_inches='tight')
    print(f"Matrice de confusion sauvegardée: ./partie_3/models/confusion_matrix_cnn.png")
    plt.show()
    
    return cm


def analyser_conflits_classes_cnn(cm, top_n=5):
    """
    Analyse les classes les plus confondues dans la matrice de confusion.
    
    Args:
        cm: matrice de confusion
        top_n: nombre de confusions à afficher
    """
    print("\n" + "="*50)
    print("--- Analyse des confusions ---")
    print("="*50)
    
    # Trouver les erreurs (hors diagonale)
    erreurs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                erreurs.append({
                    'vraie_classe': i,
                    'classe_predite': j,
                    'nombre': cm[i, j],
                    'vraie_nom': decode_class_y(i),
                    'pred_nom': decode_class_y(j)
                })
    
    # Trier par fréquence
    erreurs.sort(key=lambda x: x['nombre'], reverse=True)
    
    print(f"\nTop {top_n} confusions les plus fréquentes:\n")
    for idx, err in enumerate(erreurs[:top_n], 1):
        print(f"{idx}. {err['vraie_classe']} → {err['classe_predite']} "
              f"({err['nombre']} fois)")
        print(f"   {err['vraie_nom']} → {err['pred_nom']}\n")
    
    print("="*50 + "\n")


def comparer_cnn_vs_mlp(y_true_cnn, y_pred_cnn, y_true_mlp=None, y_pred_mlp=None):
    """
    Affiche un rapport de comparaison entre CNN et MLP (si disponible).
    
    Args:
        y_true_cnn: labels vrais CNN
        y_pred_cnn: prédictions CNN
        y_true_mlp: labels vrais MLP (optionnel)
        y_pred_mlp: prédictions MLP (optionnel)
    """
    print("\n" + "="*60)
    print("--- Rapport de Comparaison: CNN vs MLP ---")
    print("="*60)
    
    acc_cnn = accuracy_score(y_true_cnn, y_pred_cnn)
    print(f"\nAccuracy CNN:  {acc_cnn*100:.2f}%")
    
    if y_true_mlp is not None and y_pred_mlp is not None:
        acc_mlp = accuracy_score(y_true_mlp, y_pred_mlp)
        print(f"Accuracy MLP:  {acc_mlp*100:.2f}%")
        diff = (acc_cnn - acc_mlp) * 100
        symbol = "✓" if diff > 0 else "✗"
        print(f"\nDifférence:    {symbol} {abs(diff):+.2f}%")
        print(f"→ CNN est {'meilleur' if diff > 0 else 'moins bon'} que MLP")
    
    print("\nRapport de classification CNN:")
    print(classification_report(y_true_cnn, y_pred_cnn, digits=3))
    print("="*60 + "\n")
