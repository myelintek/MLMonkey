import sys
import requests
import os
from threading import Thread

from mlmonkey.constants import DATASETS_DIR


def download(url, model=None, name=None):
    if model is None:
        return False

    if name is None:
        name = url.rsplit('/', 1)[-1]

    folder = os.path.isdir(os.path.abspath(os.path.join(DATASETS_DIR, model)))
    if not folder:
        os.makedirs(folder)
    fullpath = os.path.join(folder, name)
    def run():
        with open(fullpath, "wb") as f:

            "Downloading %s" % name
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()

    Thread(target=run).start()


def extract(filename, extract_path=None):
    def run():
        file = os.path.abspath(filename)
        subname = filename.rsplit('.', 1)[-1]

        if subname in ['zip']:
            import zipfile

            zf = zipfile.ZipFile(file)

            uncompress_size = sum((file.file_size for file in zf.infolist()))

            extracted_size = 0

            for file in zf.infolist():
                extracted_size += file.file_size
                print("%s %%" % (extracted_size * 100 / uncompress_size))
                zf.extract(file)
        elif subname in ['tar', 'tgz', 'gz', 'tar.gz']:
            import tarfile
            tar = tarfile.open(file, 'r')
            for item in tar:
                tar.extract(item, extract_path)
                extract(item.name, "./" + item.name[:item.name.rfind('/')])

    Thread(target=run).start()


def verify():
    def run():
        pass
