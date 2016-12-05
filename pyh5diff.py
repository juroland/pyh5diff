import h5py
import numpy as np
import argparse
from sklearn import preprocessing

def datasets_from_file(fd):
    """ Returns all datasets (name) contained in the file described in fd """
    datasets = []
    def _append_datasets_visitor(name, item):
        if isinstance(item, h5py.Dataset):
            datasets.append(name)

    fd.visititems(_append_datasets_visitor)

    return datasets

def h5diff(filename1, filename2, precision):
    fd1 = h5py.File(filename1, 'r')
    fd2 = h5py.File(filename2, 'r')

    datasets = datasets_from_file(fd1)

    min_max_scaler = preprocessing.MinMaxScaler()

    for dataset in datasets:
        x = fd1[dataset][:].reshape(-1, 1)
        y = fd2[dataset][:].reshape(-1, 1)
        if np.issubdtype(fd1[dataset].dtype, np.integer):
            x_scaled = x
            y_scaled = y
        else:
            x_scaled = min_max_scaler.fit_transform(x)
            y_scaled = min_max_scaler.fit_transform(y)

        diff = np.fabs(x_scaled - y_scaled)
        i = np.argmax(diff)
        if  diff[i] > precision:
            print('# DIFF : (dataset: {}, id: {}'
                  ', x (file1): {}'
                  ', y (file2): {})'.format(dataset, i, x[i], y[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDF5 normalized diff')

    parser.add_argument('file1', help='file name of the first HDF5 file')
    parser.add_argument('file2', help='file name of the second HDF5 file')
    parser.add_argument('-p', '--precision', type=float, default=0,
                    help='print difference if (|a-b| > PRECISION)')

    args = parser.parse_args()

    filename1 = args.file1
    filename2 = args.file2
    precision = args.precision

    h5diff(filename1, filename2, precision)
