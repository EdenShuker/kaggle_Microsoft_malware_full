import pickle
from csv import DictReader
from time import time
import sys

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import utils

CREATE_F2V_FILE = '-create-f2v'
SAVE_MODEL = '-save'
SHOW_CONFUSION_MAT = '-show-matrix'
TRAIN = '-train'
LOAD_MODEL = '-load'
OUTPUT_FILE_FLAG = '-out-file'

# load feature
ngrams_features_list = pickle.load(open('ngrams_features'))
segments_features_set = pickle.load(open('segments_features'))


def represent_file_as_vector(dirpath, filename):
    """
    :param dirpath: path of directory that the given file is in.
    :param filename: name of file, with the extension(= .bytes or .asm) .
    :return: vector of features that represents the given file.
    """
    vec = []

    # ngrams
    curr_ngrams_set = utils.get_ngrams_set_of(dirpath, filename, n=4)
    for feature in ngrams_features_list:
        # TODO current - boolean of 'is ngram in file', optional - how many time ngrams in file
        if feature in curr_ngrams_set:
            vec.append(1)
        else:
            vec.append(0)

    # segments
    seg_counter = utils.count_seg_counts(dirpath, filename, segments_features_set)
    for seg_name in segments_features_set:
        if seg_name in seg_counter:
            vec.append(seg_counter[seg_name])
        else:
            vec.append(0)

    return vec


def create_file_file2vec(dirpath, files_list, f2v_name):
    """
    :param dirpath: path to directory that the given files are in.
    :param files_list: list of files-names.
    :param f2v_name: output file, will contain file-name and the vector that represents it.
        Format of file: filename<\t>vec
    """
    with open(f2v_name, 'w') as f:
        for f_name in files_list:
            vec = represent_file_as_vector(dirpath, f_name)  # represent each file as a vector
            vec = map(lambda x: str(x), vec)
            f.write(f_name + '\t' + ' '.join(vec) + '\n')


def get_data_and_labels(f2l_filepath, f2v_filepath):
    """
    :param f2l_filepath: path to file-to-label file (train_labels_filtered.csv) .
    :param f2v_filepath: path to file-to-vector file.
    :return: matrix and array of labels, the ith label there is connected to the ith vector in matrix.
    """
    matrix, labels = [], []

    # create file-to-label dict
    f2l_dict = utils.get_f2l_dict(f2l_filepath)

    with open(f2v_filepath, 'r') as f:
        for line in f:  # for each file find the vector that representing him
            filename, vec = line.split('\t')
            vec = map(lambda (x): int(x), vec.split(' '))

            matrix.append(vec)
            labels.append(f2l_dict[filename])
    return matrix, labels


def get_data(f2v_filepath):
    """
    :param f2v_filepath: path to file-to-vector file.
    :return: matrix which contains all the vectors inside the file.
    """
    matrix = []
    with open(f2v_filepath) as f:
        for line in f:
            _, vec = line.split('\t')
            vec = map(lambda x: int(x), vec.split(' '))
            matrix.append(vec)
    return matrix


class CodeModel(object):
    """
    Model that works with code-files, predict the type of file out of 10 possible type.
    """

    def __init__(self, lr=0.1, n_estimators=30, max_depth=5, min_child_weight=1,
                 gamma=0, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=27):
        # TODO need to tune the parameters
        self.model = xgb.XGBClassifier(learning_rate=lr,
                                       n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_child_weight=min_child_weight,
                                       gamma=gamma,
                                       subsample=subsample,
                                       colsample_bytree=colsample_bytree,
                                       scale_pos_weight=scale_pos_weight,
                                       objective='multi:softprob',
                                       seed=seed)

    def predict_on(self, matrix):
        """
        :param matrix: each row in it is a vector that represents some file.
        :return: list of labels, the ith-label is connected to the ith-vector in matrix.
        """
        preds = self.model.predict(matrix)
        return [round(val) for val in preds]

    def predict_and_accuracy_on(self, matrix, labels, show_confusion_matrix=False):
        """
        :param matrix: each row in it is a vector that represents some file.
        :param labels: list of labels, the ith-label is connected to the ith-vector in matrix.
        :param show_confusion_matrix: boolean, determine if to show to the user confusion matrix.
        :return:
        """
        # predict and find accuracy
        preds = self.predict_on(matrix)
        acc = accuracy_score(labels, preds)
        print 'accuracy %0.2f%%' % (acc * 100.0)

        # confusion matrix
        if show_confusion_matrix:
            print confusion_matrix(labels, preds)

    def train_on(self, train_tup, dev_tup, model_name=None, show_confusion_matrix=False):
        """
        Fit the model on the train-set and check its performance on dev-set, can save the model after training.
        :param train_tup: tuple of (train-matrix, labels) .
        :param dev_tup: same format as train_tup.
        :param model_name: string if needed to save the model,
            the saved model will be in file named by this string, None as default for not saving the model.
        :param show_confusion_matrix: boolean, determine if to show to the user confusion matrix.
        """
        # fit model to training data
        train_matrix, labels = train_tup
        self.model.fit(train_matrix, labels, eval_metric='mlogloss')

        # check performance of model on dev-set
        dev_matrix, labels = dev_tup
        self.predict_and_accuracy_on(dev_matrix, labels, show_confusion_matrix)

        # save model if needed
        if model_name:
            self.save_model(model_name)

    def save_model(self, filename):
        """ save the current model in a file, can be loaded from that file later. """
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load_from(filename):
        """ load a model from file """
        model = pickle.load(open(filename, 'rb'))
        return model


def train_model(args):
    """
    train a model on data, can retrain a model by loading one.
    :param args: arguments from main.
        options:
        # add '-create-f2v' to main if it is the first time to encounter the data,
                that will create a file of the given data that maps file-name to the vector representing it.
                if passed, you must pass the following arguments in the next order right after that flag:
                    dirpath f2l_filepath f2v_filepath
                explanation on each is right below.

                if not passed, then pass parameters as following:
                    f2l_filepath f2v_filepath
        # dirpath - path to directory where the files are in.
                    need to pass it only if used '-create-f2v' flag.
        # f2l_filepath - path to .csv file, each line in it is 'filename,label'.
                         must pass it.
        # f2v_filepath - path to f2v file. must pass it.
                         if you create it, so this parameter will be the name of the f2v file that
                          the program will produce.
        # add '-save' in case you want to save the model after training.
            if you pass it, you must pass right after it the name of the file that the model will be saved to.
        # add '-load' to load an existing model.
            if you pass it, you must pass right after the name of the model file to load.
    """
    if CREATE_F2V_FILE in args:  # in case of first time to process the data
        i = args.index(CREATE_F2V_FILE)
        dirpath = args[i + 1]  # path to directory where the files are in
        f2l_filepath = args[i + 2]  # path to a .csv file, each row is 'filename,label'
        f2v_filepath = args[i + 3]  # name of file to create, will contain file-name and its vector (name<\t>vec)

        # extract names of files
        file_list = []
        csv_dict = DictReader(open(f2l_filepath))
        for row in csv_dict:
            file_list.append(row['Id'])

        create_file_file2vec(dirpath, file_list, f2v_filepath)
    else:  # f2v file exists already
        f2l_filepath = args[0]
        f2v_filepath = args[1]

    # extract parameters from main
    model_name = None
    if SAVE_MODEL in args:
        model_name = args[args.index(SAVE_MODEL) + 1]
    show_conf_matrix = SHOW_CONFUSION_MAT in args

    # load data
    matrix, labels = get_data_and_labels(f2l_filepath, f2v_filepath)
    train, dev, y_train, y_dev = train_test_split(matrix, labels, test_size=0.33, random_state=42)
    dev, test, y_dev, y_test = train_test_split(dev, y_dev, test_size=0.33, random_state=43)

    # apply model
    if LOAD_MODEL in args:
        given_model_file = args[args.index(LOAD_MODEL) + 1]
        model = CodeModel.load_from(given_model_file)
    else:
        model = CodeModel()
    model.train_on((train, y_train), (dev, y_dev), model_name, show_conf_matrix)
    model.predict_and_accuracy_on(test, y_test, show_conf_matrix)


def use_exist_model(args):
    """
    use an existing model for blind prediction.
    :param args: arguments from main.
        options:
        # add '-create-f2v' to main if it is the first time to encounter the data,
                that will create a file of the given data that maps file-name to the vector representing it.
                if passed, you must pass the following arguments in the next order right after that flag:
                    dirpath files_filepath f2v_filepath
                explanation on each is right below.

                if not passed, then pass parameters as following:
                    files_filepath f2v_filepath
        # dirpath - path to directory where the files are in.
                    need to pass it only if used '-create-f2v' flag.
        # files_filepath - path to file, each line in it is a file-name.
                         must pass it.
        # f2v_filepath - path to f2v file. must pass it.
                         if you create it, so this parameter will be the name of the f2v file that
                          the program will produce.
        # add '-load' to load an existing model.
            must pass it, you must pass right after the name of the model file to load.
        # add '-out-file' and right after a name of file which will contain, after running this program,
            in each line 'filename,label' where the label is the prediction of the model.
            must pass it.
    """
    if CREATE_F2V_FILE in args:  # in case of first time to process the data
        i = args.index(CREATE_F2V_FILE)
        dirpath = args[i + 1]
        files_filepath = args[i + 2]
        f2v_filepath = args[i + 3]

        # extract names of files
        file_list = []
        with open(files_filepath) as f:
            for line in f:
                file_list.append(line)

        create_file_file2vec(dirpath, file_list, f2v_filepath)
    else:  # f2v file exists already
        files_filepath = args[0]
        f2v_filepath = args[1]

    matrix = get_data(f2v_filepath)  # load data

    # apply model
    given_model_file = args[args.index(LOAD_MODEL) + 1]
    model = CodeModel.load_from(given_model_file)
    preds = model.predict_on(matrix)

    # write output file
    output_filepath = args[args.index(OUTPUT_FILE_FLAG) + 1]
    with open(output_filepath, 'w') as out_file:
        with open(files_filepath) as input_file:
            for i, file_name in enumerate(input_file):
                out_file.write('%s,%i\n' % (file_name, preds[i]))


def main():
    """
    pass '-train' to train a model and pass the parameters needed in order to train,
    else pass the parameters needed to predict on blind-test.
    """
    print 'start'
    t0 = time()

    args = sys.argv[1:]
    if TRAIN in args:  # train a model
        train_model(args)
    else:  # blind test
        use_exist_model(args)

    print 'time to run:', time() - t0


if __name__ == '__main__':
    main()
