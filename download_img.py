from urllib.parse import urlparse
from urllib.request import urlretrieve
import os
import zipfile


def download_zip(url, zip_file):
    print('downloading...' + zip_file)
    urlretrieve(url, zip_file)

def unpack_zip(file, path):
    with zipfile.ZipFile(file, 'r') as zip:
        print('extracting...' + file)
        zip.extractall(path)

def delete_zip(file):
        print('deleting...' + file)
        os.remove(file)



def dowload_img(file):
    with open(file, 'r', encoding = "ISO-8859-1") as input:
        doc = input.read()
        token = doc.split()
        for url in token:
            a = urlparse(url)
            zip_file = 'pframe_trainvaltest/'+os.path.basename(a.path)
            download_zip(url, zip_file)
            path = 'pframe_trainvaltest/'
            unpack_zip(zip_file, path)
            delete_zip(zip_file)


dowload_img('ourkit.txt')