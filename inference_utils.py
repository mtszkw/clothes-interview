import tarfile
import numpy as np


def extract_model(model_bin_path):
    if model_bin_path.endswith("tar.gz"):
        tar = tarfile.open(model_bin_path, "r:gz")
        tar.extractall()
        tar.close()

def prediction_vector_to_label(y_pred):
    class_names = {
        0: 'dress',
        1: 'hat',
        2: 'longsleeve',
        3: 'outwear',
        4: 'pants',
        5: 'shirt',
        6: 'shoes',
        7: 'shorts',
        8: 'skirt',
        9: 't-shirt'
    }

    max_score = y_pred.max()
    max_label = np.argmax(y_pred, axis=1)[0]
    return class_names[max_label], max_score
