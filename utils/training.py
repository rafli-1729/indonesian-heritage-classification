from fastai.vision.all import MixUp, SaveModelCallback, minimum, steep, valley, slide

def find_learning_rate(learn):
    """
    Mencari learning rate optimal menggunakan lr_find.
    """
    lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    print(f'Learning rate optimal: {lrs}')
    return lrs

def train_model(learn, epochs, lr, mixup_val=0.6):
    """
    Melatih model menggunakan fit_one_cycle.

    Args:
        learn (Learner): Objek learner.
        epochs (int): Jumlah epoch.
        lr (float): Learning rate.
        mixup_val (float): Nilai alpha untuk MixUp.
    """
    callbacks = [
        MixUp(mixup_val),
        SaveModelCallback(monitor='f1_score')
    ]
    learn.fit_one_cycle(epochs, lr, cbs=callbacks)
