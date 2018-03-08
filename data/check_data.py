from collections import defaultdict
from csv import DictReader

import utils


def csv_dict_to_new_dict():
    """
    Create a dict that map label-num to list of files of the label.
    :return: dict.
    """
    csv_dict = DictReader(open('trainLabels.csv'))
    label_to_files_dict = defaultdict(list)

    for row in csv_dict:
        file_name = row['Id']
        label = row['Class']
        label_to_files_dict[label].append(file_name)
    return label_to_files_dict


def create_new_csv(l2f, f_list, add_labels_set=False):
    """
    Create new csv-file which is a filtered file of trainLabels.csv,
    will contain info about the existing files we have.
    :param l2f: dict of label-number to file-name.
    :param f_list: list of file-names.
    :param add_labels_set: bool which tell if needed to add a line of set of labels in the file.
    """
    csv_f = open('train_labels_filtered.csv', 'w')
    csv_f.write('Id,Class\n')
    labels_set = set()

    for f_name in f_list:
        # find the label of the file
        for label in l2f:
            if f_name in l2f[label]:
                csv_f.write('%s,%s\n' % (f_name, label))  # write the file and its label
                labels_set.add(label)
                break  # continue to next file

    if add_labels_set:
        labels_set = sorted(labels_set)
        csv_f.write('\nlabels: ' + ','.join(labels_set))
    csv_f.close()


if __name__ == '__main__':
    label2files = csv_dict_to_new_dict()
    path = 'train50'

    ending = '.bytes'
    malware_file_list = utils.get_files_from_dir(path, ending)
    create_new_csv(label2files, malware_file_list)
