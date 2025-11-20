import os
import shutil

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_str(arg):
    return list(map(str, arg.split(',')))

EXEC_DIR = '/home/isabela/Codes/LibIFT/bin'
RESULTS_DIR = '/home/isabela/Codes/LibIFT/Results/'
DATASETS_DIR = '/home/isabela/Codes/LibIFT/Datasets/'
ARCH_DIR = '/home/isabela/Codes/LibIFT/Datasets/archs/'

def read_dir_or_CSV(path, read_extension : bool = True):
    files = []
    if(os.path.isdir(path)):
        if(read_extension):
            files = [os.path.basename(file) for file in os.listdir(path)]
        else:
            files = [os.path.basename(file).split('.')[0] for file in os.listdir(path)]
    else:
        with open(path, 'r') as f:
            if(read_extension):
                files = [os.path.basename(file) for file in f.read().splitlines()]
            else:
                files = [os.path.basename(file).split('.')[0] for file in f.read().splitlines()]

    return files

def CheckFilesRelation_dirOrCSV(path1, path2, same_extension : bool = True):
    """
    Check if path1 and path2 contain the same files.
    Args:
        path1 (str): Path to the first directory or file containing the list of files.
        path2 (str): Path to the second directory or file containing the list of files.
    """

    if(not os.path.exists(path1)):
        raise ValueError("check_files_dirOrCSV: path1 does not exist")

    if(not os.path.exists(path2)):
        raise ValueError("check_files_dirOrCSV: path2 does not exist")

    files1 = read_dir_or_CSV(path1, read_extension = same_extension)
    files2 = read_dir_or_CSV(path2, read_extension = same_extension)

    if set(files1) > set(files2):
        return "gt"
    elif set(files1) < set(files2):
        return "lt"
    elif set(files1) == set(files2):
        return "eq"
    else:
        return "diff"

def copyGraphsAndLabels(src_dir, input_graphs_path, input_labels_path, model_dir):
    """
    Copy the graphs and labels from the source directory to the model directory.
    Args:
        src_dir (str): Source directory containing the filenames to be copied.
        input_graphs_path (str): Path to the input graphs directory.
        input_labels_path (str): Path to the input labels directory.
        model_dir (str): Destination model directory where the graphs and labels will be copied.
    """
    if(CheckFilesRelation_dirOrCSV(input_graphs_path, input_labels_path, same_extension = False) != 'eq'):
        raise ValueError("The files in input_graphs_path and input_labels_path do not match.")

    rel_src = CheckFilesRelation_dirOrCSV(src_dir, input_labels_path, same_extension = False)
    if(rel_src == 'diff' or rel_src == 'gt'):
        raise ValueError("The files in src_dir and input_labels_path do not match.")

    # remove any existing layer and label directories in the model directory
    shutil.rmtree(os.path.join(model_dir, 'layer0'), ignore_errors=True)
    shutil.rmtree(os.path.join(model_dir, 'labels0'), ignore_errors=True)

    # create the output directories
    os.makedirs(os.path.join(model_dir, 'layer0'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'labels0'), exist_ok=True)

    if(os.path.isdir(src_dir)):
        # read the files from the directory
        filenames = os.listdir(src_dir)
    else:
        # read the files paths from a csv file
        with open(src_dir, 'r') as f:
            # Remove any leading/trailing whitespace characters and the only the filenames
            filenames = [f.strip().split('/')[-1] for f in f.read().splitlines()]

    for file_name in filenames:

        # get the file name
        if file_name.endswith('-seeds.txt'):
            graph_name = file_name.split('-')[0]
        else:
            graph_name = file_name.split('.')[0]

        # create symlinks for the graph and label files
        graph_src_file = os.path.join(input_graphs_path, f'{graph_name}.json')
        os.symlink(graph_src_file, os.path.join(model_dir, 'layer0', f'{graph_name}.json'))

        labels_src_file = os.path.join(input_labels_path, f'{graph_name}.pgm')
        os.symlink(labels_src_file, os.path.join(model_dir, 'labels0', f'{graph_name}.pgm'))

        