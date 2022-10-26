import sys
import re
import pickle

FILENAMES = ["train.txt"]

target_names = [
    "==Poor==", 
    "==Unsatisfactory==", 
    "==Good==", 
    "==VeryGood==", 
    "==Excellent=="
    ]

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        FILENAMES = sys.argv[1:]

for filename in FILENAMES:

    train = {"data": [], "target": [], "target_names": target_names}

    file = open(filename, 'r')
    Lines = file.readlines()
    
    for line in Lines:
        target_name, text = re.split('\t', line, maxsplit=1)
        text = text[:-4] # removing '\t\t\t\n' from line ends
        train["data"].append(text)
        target = target_names.index(target_name)
        train["target"].append(target)

    new_filename = filename.split(".")[1][1:]+".pkl"
    dict_file = open(new_filename, "wb")
    pickle.dump(train, dict_file)
    dict_file.close()
