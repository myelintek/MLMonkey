import sys
import requests
import os
from threading import Thread

from mlmonkey.constants import DATASETS_DIR
from mlmonkey.log import logger


def download(url, model=None, assign_path=None, name=None):
    if model is None:
        return False

    if name is None:
        name = url.rsplit('/', 1)[-1]

    if assign_path:
        folder = os.path.isdir(os.path.abspath(assign_path))
    else:
        folder = os.path.isdir(os.path.abspath(os.path.join(DATASETS_DIR, model)))

    if not folder:
        os.makedirs(folder)

    fullpath = os.path.join(folder, name)

    def run():
        with open(fullpath, "wb") as f:

            logger.debug("Downloading %s..." % name)
            try:
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
            except Exception as e:
                logger.warning("Downloading %s fail." % name)
                logger.warning(str(e))

    Thread(target=run).start()
    return fullpath


def extract(filename, extract_path=None):
    def run():

        logger.debug("Extracting %s..." % filename)
        file = os.path.abspath(filename)
        subname = filename.rsplit('.', 1)[-1]

        if subname in ['zip']:
            import zipfile
            try:
                zf = zipfile.ZipFile(file)
                uncompress_size = sum((file.file_size for file in zf.infolist()))
                extracted_size = 0
                for file in zf.infolist():
                    extracted_size += file.file_size
                    logger.debug("%s %%" % (extracted_size * 100 / uncompress_size))
                    zf.extract(file)
            except Exception as e:
                logger.warning("Unzip %s file fail." % filename)
                logger.warning(str(e))

        elif subname in ['tar', 'tgz', 'gz', 'tar.gz']:
            import tarfile
            try:
                tar = tarfile.open(file, 'r')
                for item in tar:
                    tar.extract(item, extract_path)
                    extract(item.name, "./" + item.name[:item.name.rfind('/')])
            except Exception as e:
                logger.warning("Extract %s tarfile fail." % filename)

    Thread(target=run).start()
