from test.training_set_name import path
import voter_turnout.io as io

if __name__ == '__main__':
    df = io.readFile(path)

    print("All tests passed")
