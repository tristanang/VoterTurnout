from test.training_set_name import path
import voter_turnout.io as io



if __name__ == '__main__':
    df = io.readFile(path)

    assert df.size #makes sure df actually exists

    print("This is the number of columns before dropping: " + str(df.columns.size))

    df = io.dropSameColumn(df)

    print("This is the number of columns after dropping: " + str(df.columns.size))

    df = io.oneHot(df)

    print("This is the number of columns after one hotting: " + str(df.columns.size))

    print("All tests passed")
