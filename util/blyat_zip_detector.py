import os
import zipfile

def is_zip_file(filename):
    try:
        with open(filename, 'rb') as f:
            return zipfile.is_zipfile(f)
    except Exception:
        return False

def check_for_zip_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if is_zip_file(filepath):
                print(f"Файл ZIP найден: {filepath}")

check_for_zip_files('res')