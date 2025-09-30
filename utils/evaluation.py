import pandas as pd
from fastai.vision.all import ClassificationInterpretation, get_image_files

def plot_results(learn):
    """
    Menampilkan confusion matrix dan top losses.
    """
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(9, figsize=(15, 10))
    interp.print_classification_report()

def create_submission(learn, test_path, dls, use_tta=False, filename='submission.csv'):
    """
    Membuat file submission CSV dari hasil prediksi pada data tes.

    Args:
        learn (Learner): Learner yang sudah dilatih.
        test_path (Path): Path ke data tes.
        dls (DataLoaders): DataLoaders yang digunakan (untuk vocab).
        use_tta (bool): Apakah menggunakan Test Time Augmentation.
        filename (str): Nama file CSV untuk submission.
    """
    test_files = get_image_files(test_path)
    test_dl = dls.test_dl(test_files)
    
    if use_tta:
        print("Menggunakan Test Time Augmentation (TTA)...")
        preds, _ = learn.tta(dl=test_dl)
    else:
        print("Menggunakan get_preds...")
        preds, _ = learn.get_preds(dl=test_dl)
        
    pred_labels = preds.argmax(dim=1)
    label_names = [dls.vocab[i] for i in pred_labels]
    
    submission_df = pd.DataFrame({
        'id': [f.name[:-4] for f in test_files],
        'style': label_names
    })
    
    submission_df.sort_values('id', inplace=True)
    submission_df.to_csv(filename, index=False)
    print(f"File submission berhasil disimpan sebagai '{filename}'")
    print("\\nDistribusi kelas pada file submission:")
    print(submission_df['style'].value_counts())
