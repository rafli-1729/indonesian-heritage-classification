import hashlib
import shutil
from pathlib import Path
from fastai.vision.all import get_image_files, progress_bar

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
