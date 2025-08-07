#!/usr/bin/env python3
import urllib.request
import zipfile
import os

def main():
    # Download buffalo_l model
    model_dir = '/root/.insightface/models/buffalo_l'
    os.makedirs(model_dir, exist_ok=True)
    zip_path = '/root/.insightface/models/buffalo_l.zip'
    url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'

    print('Downloading buffalo_l model...')
    urllib.request.urlretrieve(url, zip_path)

    print('Extracting buffalo_l model...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/root/.insightface/models/')

    # Clean up zip file
    os.remove(zip_path)
    print('buffalo_l model downloaded and extracted successfully')

if __name__ == '__main__':
    main()