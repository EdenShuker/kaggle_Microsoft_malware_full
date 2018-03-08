import pickle

import utils


def get_seg_set_of_file(dirpath, filepath):
    """
    :param dirpath: path to directory.
    :param filepath: path to file in the directory.
    :return: set of segments from the given file.
    """
    seg_set = set()
    with open('%s/%s.asm' % (dirpath, filepath)) as f:
        for line in f:
            segment_name = line.split(':', 1)[0]
            seg_set.add(segment_name)
    return seg_set


def get_segment_set_of(dirpath):
    """
    :param dirpath: path to directory.
    :return: set of segments-names extracted from all the files in the given directory.
    """
    asm_files = utils.get_files_from_dir(dirpath, '.asm')  # get list of files
    seg_set = set()

    for asm_f in asm_files:
        seg_set.update(get_seg_set_of_file(dirpath, asm_f))
    return seg_set


if __name__ == '__main__':
    segments = get_segment_set_of('train50')
    segments.add(utils.UNK)
    print segments
    pickle.dump(segments, open('segments_names', 'w'))
