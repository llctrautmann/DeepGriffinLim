import os
import glob

def remove_files(directory, extensions):
    for extension in extensions:
        for filepath in glob.iglob(directory + '**/*.' + extension, recursive=True):
            if "data" not in filepath:  # skip files in the data folder
                print(filepath)
                os.remove(filepath)

directory = '/path/to/your/directory'  # replace with your directory
extensions = ['png', 'wav']

if __name__ == '__main__':
    directory = './out/' # replace with your directory
    extensions = ['png', 'wav']

    remove_files(directory=directory, extensions=extensions)

    print('Done!')