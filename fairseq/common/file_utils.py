"""
Utilities for working with the local dataset cache.
"""

import os
import logging
import shutil
import tempfile
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set
from hashlib import sha256
from functools import wraps

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # pylint: disable=import-error

from fairseq.common.tqdm import Tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CACHE_ROOT = Path(os.getenv('FAIRSEQ_CACHE_ROOT', Path.home() / '.fairseq'))
CACHE_DIRECTORY = str(CACHE_ROOT / "cache")
DEPRECATED_CACHE_DIRECTORY = str(CACHE_ROOT / "datasets")

# This variable was deprecated in 0.7.2 since we use a single folder for caching
# all types of files (datasets, models, etc.)
DATASET_CACHE = CACHE_DIRECTORY

# Warn if the user is still using the deprecated cache directory.
if os.path.exists(DEPRECATED_CACHE_DIRECTORY):
  logger = logging.getLogger(__name__) # pylint: disable=invalid-name
  logger.warning(f"Deprecated cache directory found ({DEPRECATED_CACHE_DIRECTORY}).  "
                  f"Please remove this directory from your system to free up space.")


def url_to_filename(url: str, etag: str = None) -> str:
  """
  Convert `url` into a hashed filename in a repeatable way.
  If `etag` is specified, append its hash to the url's, delimited
  by a period.
  """
  url_bytes = url.encode('utf-8')
  url_hash = sha256(url_bytes)
  filename = url_hash.hexdigest()

  if etag:
    etag_bytes = etag.encode('utf-8')
    etag_hash = sha256(etag_bytes)
    filename += '.' + etag_hash.hexdigest()

  return filename


def filename_to_url(filename: str, cache_dir: str = None) -> Tuple[str, str]:
  """
  Return the url and etag (which may be ``None``) stored for `filename`.
  Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.
  """
  if cache_dir is None:
    cache_dir = CACHE_DIRECTORY

  cache_path = os.path.join(cache_dir, filename)
  if not os.path.exists(cache_path):
    raise FileNotFoundError("file {} not found".format(cache_path))

  meta_path = cache_path + '.json'
  if not os.path.exists(meta_path):
    raise FileNotFoundError("file {} not found".format(meta_path))

  with open(meta_path) as meta_file:
    metadata = json.load(meta_file)
  url = metadata['url']
  etag = metadata['etag']

  return url, etag

def cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str:
  """
  Given something that might be a URL (or might be a local path),
  determine which. If it's a URL, download the file and cache it, and
  return the path to the cached file. If it's already a local path,
  make sure the file exists and then return the path.
  """
  if cache_dir is None:
    cache_dir = CACHE_DIRECTORY
  if isinstance(url_or_filename, Path):
    url_or_filename = str(url_or_filename)

  url_or_filename = os.path.expanduser(url_or_filename)
  parsed = urlparse(url_or_filename)

  if parsed.scheme in ('http', 'https'):
    # URL, so get it from the cache (downloading if necessary)
    return get_from_cache(url_or_filename, cache_dir)
  elif os.path.exists(url_or_filename):
    # File, and it exists.
    return url_or_filename
  elif parsed.scheme == '':
    # File, but it doesn't exist.
    raise FileNotFoundError("file {} not found".format(url_or_filename))
  else:
    # Something unknown
    raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

def is_url_or_existing_file(url_or_filename: Union[str, Path, None]) -> bool:
  """
  Given something that might be a URL (or might be a local path),
  determine check if it's url or an existing file path.
  """
  if url_or_filename is None:
    return False
  url_or_filename = os.path.expanduser(str(url_or_filename))
  parsed = urlparse(url_or_filename)
  return parsed.scheme in ('http', 'https') or os.path.exists(url_or_filename)


def session_with_backoff() -> requests.Session:
  """
  We ran into an issue where http requests to s3 were timing out,
  possibly because we were making too many requests too quickly.
  This helper function returns a requests session that has retry-with-backoff
  built in.
  see stackoverflow.com/questions/23267409/how-to-implement-retry-mechanism-into-python-requests-library
  """
  session = requests.Session()
  retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
  session.mount('http://', HTTPAdapter(max_retries=retries))
  session.mount('https://', HTTPAdapter(max_retries=retries))

  return session

def http_get(url: str, temp_file: IO) -> None:
  with session_with_backoff() as session:
    req = session.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = Tqdm.tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
      if chunk: # filter out keep-alive new chunks
        progress.update(len(chunk))
        temp_file.write(chunk)
    progress.close()


# TODO(joelgrus): do we want to do checksums or anything like that?
def get_from_cache(url: str, cache_dir: str = None) -> str:
  """
  Given a URL, look for the corresponding dataset in the local cache.
  If it's not there, download it. Then return the path to the cached file.
  """
  if cache_dir is None:
    cache_dir = CACHE_DIRECTORY

  os.makedirs(cache_dir, exist_ok=True)

  # Get eTag to add to filename, if it exists.
  if url.startswith("s3://"):
    etag = s3_etag(url)
  else:
    with session_with_backoff() as session:
      response = session.head(url, allow_redirects=True)
    if response.status_code != 200:
      raise IOError("HEAD request failed for url {} with status code {}"
                    .format(url, response.status_code))
    etag = response.headers.get("ETag")

  filename = url_to_filename(url, etag)

  # get cache path to put the file
  cache_path = os.path.join(cache_dir, filename)

  if not os.path.exists(cache_path):
    # Download to temporary file, then copy to cache dir once finished.
    # Otherwise you get corrupt cache entries if the download gets interrupted.
    with tempfile.NamedTemporaryFile() as temp_file:
      logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

      # GET file object
      http_get(url, temp_file)

      # we are copying the file before closing it, so flush to avoid truncation
      temp_file.flush()
      # shutil.copyfileobj() starts at the current position, so go to the start
      temp_file.seek(0)

      logger.info("copying %s to cache at %s", temp_file.name, cache_path)
      with open(cache_path, 'wb') as cache_file:
        shutil.copyfileobj(temp_file, cache_file)

      logger.info("creating metadata file for %s", cache_path)
      meta = {'url': url, 'etag': etag}
      meta_path = cache_path + '.json'
      with open(meta_path, 'w') as meta_file:
        json.dump(meta, meta_file)

      logger.info("removing temp file %s", temp_file.name)

  return cache_path


def read_set_from_file(filename: str) -> Set[str]:
  """
  Extract a de-duped collection (set) of text from a file.
  Expected file format is one item per line.
  """
  collection = set()
  with open(filename, 'r') as file_:
      for line in file_:
          collection.add(line.rstrip())
  return collection


def get_file_extension(path: str, dot=True, lower: bool = True):
  ext = os.path.splitext(path)[1]
  ext = ext if dot else ext[1:]
  return ext.lower() if lower else ext
