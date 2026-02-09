
import zipfile
import os

zip_files = [f for f in os.listdir('chat_history') if f.endswith('.zip')]

for zip_file in zip_files:
    print(f"Inspecting {zip_file}...")
    with zipfile.ZipFile(os.path.join('chat_history', zip_file), 'r') as z:
        for file_info in z.infolist():
            if file_info.filename.endswith('.txt'):
                print(f"Found text file: {file_info.filename} ({file_info.file_size} bytes)")
