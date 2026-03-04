import tensorflow as tf

def load_image_cnn_et_preparation(path, label, augment=False):
    """
    Charge une image et l'augmente SANS l'aplatir.
    La shape reste (H, W, C) pour être compatible avec les couches Conv2D.
    
    Args:
        path: chemin de l'image
        label: classe (entier)
        augment: booléen, activer augmentation
        
    Returns:
        (img, label) où img a shape (H, W, C)
    """
    # Lecture
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)

    # Normalisation
    img = tf.cast(img, tf.float32) / 255.0

    # Augmentation modérée
    if augment:
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_flip_left_right(img)

    # Important: Ne PAS aplatir pour CNN
    # Shape reste (H, W, C) ex: (64, 64, 3)
    y = tf.cast(label, tf.int32)
    return img, y


def construire_dataset_cnn(couples, batch_size=64, shuffle=True, augment=False):
    """
    Construit un dataset TensorFlow pour CNN (pas d'aplatissement).
    Les images restent en shape (H, W, C).
    
    Args:
        couples: liste de tuples (path, class_id)
        batch_size: taille des batches
        shuffle: mélanger les données
        augment: appliquer augmentation
        
    Returns:
        tf.data.Dataset avec images (H, W, C) et labels
    """
    paths = [p for p, _ in couples]
    labels = [c for _, c in couples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: load_image_cnn_et_preparation(p, l, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def decode_class_y(y):
    """
    Convertit un id de classe en nom lisible (string)
    """
    class_names = {
        0: "speed limit 20",
        1: "speed limit 30",
        2: "speed limit 50",
        3: "speed limit 60",
        4: "speed limit 70",
        5: "speed limit 80",
        6: "restriction ends 80",
        7: "speed limit 100",
        8: "speed limit 120",
        9: "no overtaking",
        10: "no overtaking trucks",
        11: "right of way next intersection",
        12: "right of way",
        13: "yield",
        14: "stop",
        15: "no entry",
        16: "forbidden trucks",
        17: "max weight",
        18: "max width",
        19: "max height",
        20: "max axle load",
        21: "speed limit 30 zone",
        22: "speed limit 30 zone end",
        23: "speed limit 50 zone",
        24: "speed limit 50 zone end",
        25: "speed limit 60 zone",
        26: "speed limit 60 zone end",
        27: "speed limit 80 zone",
        28: "speed limit 80 zone end",
        29: "speed limit 100 zone",
        30: "speed limit 100 zone end",
        31: "speed limit 120 zone",
        32: "speed limit 120 zone end",
        33: "end of speed limits",
        34: "speed limit 80 end",
        35: "speed limit 100 end",
        36: "speed limit 120 end",
        37: "no parking",
        38: "no stopping",
        39: "one way",
        40: "walking and cycling",
        41: "walking",
        42: "cycling"
    }
    return class_names.get(y, f"classe {y}")
