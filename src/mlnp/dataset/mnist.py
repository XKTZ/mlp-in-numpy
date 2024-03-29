"""
MIT License

Copyright (c) 2017 Hyeonseok Jung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from mlnp.device import device as np, default_cont_type
from urllib import request
import gzip
import pickle

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist(path):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], f"{path}/{name[1]}")
    print("Download complete.")


def save_mnist(path):
    mnist = {}
    for name in filename[:2]:
        with gzip.open(f"{path}/{name[1]}", 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    for name in filename[-2:]:
        with gzip.open(f"{path}/{name[1]}", 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(f"{path}/mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init(path):
    download_mnist(path)
    save_mnist(path)


def load(path, dtype = default_cont_type):
    with open(f"{path}/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"].astype(dtype), \
        mnist["training_labels"].astype(int), \
        mnist["test_images"].astype(dtype), \
        mnist["test_labels"].astype(int)
