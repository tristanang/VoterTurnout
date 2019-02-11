from test import training_set_name
from voter_turnout.preprocess import io, gradient

def main(filename, path):
    df = io.main(path)
    df = gradient.main(df)

    savePath = 'data/' + filename[:-4] + '/input.pickle'

    df.to_pickle(savePath)

if __name__ == '__main__':
    for i in range(len(training_set_name.filenames)):
        main(training_set_name.filenames[i], training_set_name.paths[i])
