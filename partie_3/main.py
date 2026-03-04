from partie_1.methode_de_base import load_json
from partie_1.Filtrage_des_classes.Pretraitement import filtre_imagettes_selon_liste
from partie_3.Conception_du_CNN.build_cnn import build_cnn_model, compile_model
from partie_3.Entrainement.train_cnn import (
    entrainer_cnn, 
    tracer_courbes_cnn, 
    save_model_cnn, 
    save_history_cnn, 
    load_model_cnn
)
from partie_3.Evaluation.evaluation_cnn import (
    evaluer_performances_cnn, 
    get_predictions_cnn, 
    afficher_matrice_confusion_cnn, 
    analyser_conflits_classes_cnn,
    comparer_cnn_vs_mlp
)
from partie_3.Preparation_des_donnees.prepare_data_cnn import construire_dataset_cnn
import time


# Chemins des données
imagettes_dir_64x64 = "./data/64x64/"
annotation_file = "./data/gt.txt"

json_couple_path_class_train = imagettes_dir_64x64 + "train.json"
json_couple_path_class_val = imagettes_dir_64x64 + "val.json"
json_couple_path_class_test = imagettes_dir_64x64 + "test.json"


def entrainement_cnn_complet(epochs=30, batch_size=64, size_image=64):
    """
    Entraînement complet du modèle CNN.
    
    Architecture:
    - Conv2D(32) + ReLU + MaxPool
    - Conv2D(64) + ReLU + MaxPool
    - Conv2D(128) + ReLU
    - Flatten
    - Dense(128) + ReLU + Dropout(0.5)
    - Dense(nb_classes) + Softmax
    
    Augmentation: brightness, contrast, flip
    Loss: sparse_categorical_crossentropy
    Optimiser: Adam
    """
    print("\n" + "="*60)
    print("PARTIE 3 - ENTRAÎNEMENT CNN")
    print("="*60)
    
    # Chargement des données
    print("\n[1/5] Chargement des données...")
    train = load_json(json_couple_path_class_train)
    val = load_json(json_couple_path_class_val)
    
    # Sélection des classes
    class_selectioner = filtre_imagettes_selon_liste(imagettes_dir_64x64, annotation_file)
    nb_classes = len(class_selectioner)
    print(f"    → {len(train)} images d'entraînement")
    print(f"    → {len(val)} images de validation")
    print(f"    → {nb_classes} classes sélectionnées")
    
    # Construction des datasets
    print("\n[2/5] Construction des datasets (sans aplatir)...")
    train_dataset = construire_dataset_cnn(train, batch_size=batch_size, shuffle=True, augment=True)
    val_dataset = construire_dataset_cnn(val, batch_size=batch_size, shuffle=True, augment=False)
    print("    → Datasets créés avec augmentation active")
    
    # Construction du modèle
    print("\n[3/5] Construction du modèle CNN...")
    model = build_cnn_model(num_classes=nb_classes, input_shape=(size_image, size_image, 3))
    model = compile_model(model, learning_rate=0.001)
    print(f"    → Modèle compilé, {model.count_params()} paramètres")
    model.summary()
    
    # Entraînement
    print("\n[4/5] Entraînement du modèle...")
    start = time.time()
    model, history = entrainer_cnn(model, train_dataset, val_dataset, epochs=epochs)
    end = time.time()
    print(f"    → Temps d'entraînement: {end - start:.2f} secondes ({(end-start)/60:.2f} min)")
    
    # Tracer les courbes
    print("\n[5/5] Sauvegarde et visualisation...")
    tracer_courbes_cnn(history)
    save_model_cnn(model)
    save_history_cnn(history)
    
    print("\n" + "="*60)
    print("✓ Entraînement CNN terminé")
    print("="*60 + "\n")
    
    return model, history


def test_cnn_complet(batch_size=64, model_path="./partie_3/models/CNN_64x64.keras"):
    """
    Test complet du modèle CNN entraîné.
    Chargement, évaluation, matrices de confusion, comparaison.
    """
    print("\n" + "="*60)
    print("PARTIE 3 - TEST ET ÉVALUATION CNN")
    print("="*60)
    
    # Chargement des données
    print("\n[1/4] Chargement des données...")
    train = load_json(json_couple_path_class_train)
    test = load_json(json_couple_path_class_test)
    
    # Construction des datasets
    print("[2/4] Construction des datasets...")
    train_ds = construire_dataset_cnn(train, batch_size=batch_size, shuffle=False, augment=False)
    test_ds = construire_dataset_cnn(test, batch_size=batch_size, shuffle=False, augment=False)
    
    # Chargement du modèle
    print("[3/4] Chargement du modèle CNN...")
    model = load_model_cnn(model_path)
    if model is None:
        print("❌ Impossible de charger le modèle. Lancez d'abord: entrainement_cnn_complet()")
        return
    
    # Évaluation
    print("[4/4] Évaluation des performances...")
    evaluer_performances_cnn(model, train_ds, test_ds)
    
    # Récupération des prédictions
    print("Récupération des prédictions (test)...")
    y_true, y_pred = get_predictions_cnn(model, test_ds)
    
    # Matrice de confusion
    cm = afficher_matrice_confusion_cnn(y_true, y_pred, title="Matrice de confusion - CNN (Test)")
    
    # Analyse des confusions
    analyser_conflits_classes_cnn(cm)
    
    # Comparaison avec MLP (optionnel)
    print("\n💡 Pour comparer avec le MLP, charger les prédictions MLP depuis partie_2.")
    
    print("="*60)
    print("✓ Évaluation CNN terminée")
    print("="*60 + "\n")


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    # Commenter/décommenter selon votre choix
    
    # 1. Entraîner le modèle CNN
     #entrainement_cnn_complet(epochs=30, batch_size=64)
    
    # 2. Tester le modèle CNN
    test_cnn_complet()
