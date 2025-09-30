import torch
import os
import numpy as np
from fastai.vision.all import (
    vision_learner, LabelSmoothingCrossEntropy, FocalLoss, F1Score
)

def get_class_weights(dls, train_path):
    """
    Menghitung bobot untuk setiap kelas berdasarkan jumlah sampel.
    """
    class_names = sorted([d.name for d in os.scandir(train_path) if d.is_dir()])
    counts = np.array([len(os.listdir(train_path/c)) for c in class_names])
    weights = (counts.sum() / counts)
    weights_tensor = torch.FloatTensor(weights).to(dls.device)
    
    print("Bobot untuk setiap kelas:")
    for class_name, weight in zip(class_names, weights):
        print(f"- {class_name}: {weight:.2f}")
    return weights_tensor

def create_learner(dls, model_arch, loss_func_name='FocalLoss', metrics=F1Score(average='macro'), path='/kaggle/working/'):
    """
    Membuat vision_learner dari Fastai.

    Args:
        dls (DataLoaders): DataLoaders yang akan digunakan.
        model_arch (str): Nama arsitektur model (misal: 'tf_efficientnetv2_m.in21k').
        loss_func_name (str): Nama loss function ('FocalLoss' atau 'LabelSmoothingCrossEntropy').
        metrics: Metrik evaluasi.
        path (str): Path untuk menyimpan model.
    
    Returns:
        Learner: Objek learner Fastai.
    """
    if loss_func_name == 'LabelSmoothingCrossEntropy':
        weights_tensor = get_class_weights(dls, dls.path)
        loss_func = LabelSmoothingCrossEntropy(weight=weights_tensor, eps=0.1)
    else:
        loss_func = FocalLoss()
        
    learn = vision_learner(
        dls,
        model_arch,
        metrics=metrics,
        loss_func=loss_func,
        path=path
    ).to_fp16()
    
    return learn
