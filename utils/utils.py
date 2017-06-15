import os
import sys
import tarfile
from six.moves import urllib

def maybe_download_and_extract(data_url, dest_dir, file_path):
    """Download and extract the tarball from Alex's website.
    Args:
        data_url: url for dataset to be downloaded
        dest_dir: destination directory to download the dataset
        file_path: path to ~
    Returns:
        nothing
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_dir, file_path)

    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_dir)
