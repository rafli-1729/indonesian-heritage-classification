import hashlib
import shutil
from pathlib import Path
from fastai.vision.all import get_image_files, progress_bar

move_map = {
    'balinese_train_000003.jpg': 'javanese',
    'balinese_train_000066.jpg': 'javanese',
    'balinese_train_000157.jpg': 'delete',
    'balinese_train_000083.jpg': 'batak',
    'balinese_train_000192.jpg': 'delete',
    'balinese_train_000233.jpg': 'delete',
    'balinese_train_000245.jpg': 'minangkabau',
    'balinese_train_000278.jpg': 'javanese',
    'balinese_train_000341.jpg': 'delete',
    'balinese_train_000391.jpg': 'delete',
    'balinese_train_000425.jpg': 'delete',
    'balinese_train_000489.jpg': 'delete',
    'balinese_train_000529.jpg': 'javanese',
    'balinese_train_000559.jpg': 'delete',
    'balinese_train_000586.jpg': 'javanese',
    'balinese_train_000595.jpg': 'dayak',
    'balinese_train_000651.jpg': 'javanese',
    'balinese_train_000721.jpg': 'delete',
    'balinese_train_000740.jpg': 'delete',
    'balinese_train_000745.jpg': 'javanese',
    'balinese_train_000760.jpg': 'batak',
    'balinese_train_000761.jpg': 'batak',
    'balinese_train_000762.jpg': 'batak',
    'balinese_train_000763.jpg': 'dayak',
    'balinese_train_000764.jpg': 'minangkabau',
    'balinese_train_000765.jpg': 'javanese',
    'balinese_train_000766.jpg': 'javanese',
    'balinese_train_000767.jpg': 'javanese',
    'balinese_train_000768.jpg': 'javanese',
    'balinese_train_000769.jpg': 'javanese',
    'balinese_train_000770.jpg': 'javanese',
    'balinese_train_000771.jpg': 'minangkabau',
    'balinese_train_000772.jpg': 'minangkabau',
    'balinese_train_000773.jpg': 'minangkabau',
    'balinese_train_000774.jpg': 'minangkabau',
    'balinese_train_000775.jpg': 'delete',
    'balinese_train_000776.jpg': 'minangkabau',
    'batak_train_000001.jpg': 'balinese',
    'batak_train_000002.jpg': 'balinese',
    'batak_train_000003.jpg': 'delete',
    'batak_train_000004.jpg': 'delete',
    'batak_train_000007.jpg': 'balinese',
    'batak_train_000010.jpg': 'javanese',
    'batak_train_000023.jpg': 'minangkabau',
    'batak_train_000032.jpg': 'delete',
    'batak_train_000033.jpg': 'delete',
    'batak_train_000035.jpg': 'delete',
    'batak_train_000038.jpg': 'delete',
    'batak_train_000040.jpg': 'delete',
    'batak_train_000042.jpg': 'delete',
    'batak_train_000043.jpg': 'delete',
    'batak_train_000057.jpg': 'delete',
    'batak_train_000056.jpg': 'minangkabau',
    'batak_train_000059.jpg': 'minangkabau',
    'batak_train_000062.jpg': 'minangkabau',
    'batak_train_000067.jpg': 'minangkabau',
    'batak_train_000071.jpg': 'balinese',
    'batak_train_000072.jpg': 'delete',
    'batak_train_000086.jpg': 'balinese',
    'batak_train_000091.jpg': 'dayak',
    'batak_train_000092.jpg': 'javanese',
    'batak_train_000093.jpg': 'javanese',
    'batak_train_000094.jpg': 'minangkabau',
    'batak_train_000095.jpg': 'minangkabau',
    'dayak_train_000004.jpg': 'minangkabau',
    'dayak_train_000019.jpg': 'balinese',
    'dayak_train_000025.jpg': 'delete',
    'dayak_train_000026.jpg': 'minangkabau',
    'dayak_train_000052.jpg': 'javanese',
    'javanese_train_000007.jpg': 'balinese',
    'javanese_train_000009.jpg': 'balinese',
    'javanese_train_000015.jpg': 'dayak',
    'javanese_train_000024.jpg': 'balinese',
    'javanese_train_000032.jpg': 'balinese',
    'javanese_train_000050.jpg': 'balinese',
    'javanese_train_000055.jpg': 'delete',
    'javanese_train_000077.jpg': 'balinese',
    'javanese_train_000111.jpg': 'batak',
    'javanese_train_000122.jpg': 'balinese',
    'javanese_train_000202.jpg': 'balinese',
    'javanese_train_000213.jpg': 'batak',
    'javanese_train_000245.jpg': 'minangkabau',
    'javanese_train_000249.jpg': 'minangkabau',
    'minangkabau_train_000003.jpg': 'balinese',
    'minangkabau_train_000005.jpg': 'balinese',
    'minangkabau_train_000008.jpg': 'delete',
    'minangkabau_train_000010.jpg': 'balinese',
    'minangkabau_train_000012.jpg': 'batak',
    'minangkabau_train_000015.jpg': 'dayak',
    'minangkabau_train_000017.jpg': 'javanese',
    'minangkabau_train_000021.jpg': 'javanese',
    'minangkabau_train_000038.jpg': 'balinese',
    'minangkabau_train_000066.jpg': 'balinese',
    'minangkabau_train_000071.jpg': 'javanese',
    'minangkabau_train_000074.jpg': 'javanese',
    'minangkabau_train_000114.jpg': 'delete',
    'minangkabau_train_000117.jpg': 'balinese',
    'minangkabau_train_000134.jpg': 'minangkabau',
    'minangkabau_train_000180.jpg': 'javanese',
    'minangkabau_train_000200.jpg': 'javanese',
    'minangkabau_train_000231.jpg': 'delete',
    'minangkabau_train_000299.jpg': 'balinese',
    'minangkabau_train_000388.jpg': 'balinese',
    'minangkabau_train_000411.jpg': 'batak',
    'minangkabau_train_000414.jpg': 'javanese',
    'minangkabau_train_000416.jpg': 'javanese',
    'minangkabau_train_000459.jpg': 'javanese'
}

def find_and_remove_duplicates(data_path: Path):
    """
    Mencari dan menghapus file duplikat dalam direktori data.

    Args:
        data_path (Path): Path ke direktori data yang akan dibersihkan.
    """
    hashes = {}
    duplicates_to_delete = []

    def file_hash(filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    for filepath in progress_bar(get_image_files(data_path)):
        current_hash = file_hash(filepath)
        if current_hash in hashes:
            duplicates_to_delete.append(filepath.name)
        else:
            hashes[current_hash] = filepath

    print(f"\\nIdentifikasi selesai. Ditemukan {len(duplicates_to_delete)} file duplikat.")

    set_of_duplicates = set(duplicates_to_delete)
    deleted_count = 0
    
    for filename in set_of_duplicates:
        source_path = next(data_path.rglob(filename), None)
        if source_path:
            source_path.unlink()
            print(f"Deleted (Duplicate): {filename}")
            deleted_count += 1
            
    print(f"Total file duplikat yang dihapus: {deleted_count}")
    return set(duplicates_to_delete)


def fix_wrong_labels(data_path: Path, move_map: dict, duplicates: set):
    """
    Memindahkan file yang salah label ke folder yang benar.

    Args:
        data_path (Path): Path ke direktori data.
        move_map (dict): Dictionary pemetaan nama file ke folder target.
        duplicates (set): Set nama file duplikat untuk diabaikan.
    """
    moved_count = 0
    deleted_count = 0

    for filename, target_folder in move_map.items():
        if filename in duplicates:
            source_path = next(data_path.rglob(filename), None)
            if source_path:
                source_path.unlink()
                print(f"Deleted (Wrong label & Duplicate): {filename}")
                deleted_count += 1
                duplicates.remove(filename)
        else:
            source_path = next(data_path.rglob(filename), None)
            if source_path:
                destination_path = data_path / target_folder / filename
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(destination_path))
                print(f"Moved (Wrong label): {filename} ke {target_folder}")
                moved_count += 1
    
    print(f"\\nTotal file salah label yang dipindahkan: {moved_count}")
    print(f"Total file salah label & duplikat yang dihapus: {deleted_count}")
