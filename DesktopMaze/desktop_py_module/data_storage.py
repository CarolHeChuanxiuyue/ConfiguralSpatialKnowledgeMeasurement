import csv
from pathlib import Path
import numpy as np


class DataStorage:

    @staticmethod
    def get_all_files_with_suffix(directory, file_suffix):
        if not file_suffix.startswith("."):
            file_suffix = "." + file_suffix
        return [x for x in list(Path(directory).glob("*{}".format(file_suffix)))]

    @staticmethod
    def write_to_csv(filename, column_names, data_generator, delimiter=","):
        with open(filename, 'w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=delimiter)
            writer.writerow(column_names)
            for row in data_generator:
                writer.writerow(row)
            pass

    @staticmethod
    def get_array(file):
        return np.loadtxt(file, delimiter=",", skiprows=1)
        pass

    @staticmethod
    def read_from_csv(file, delimiter=","):
        with open(file, newline="") as input_file:
            reader = csv.reader(input_file, delimiter=delimiter)
            for row in reader:
                yield [float(x) for x in row]

