from test.training_set_name import path
from voter_turnout import io, info



if __name__ == '__main__':
    df = io.readFile(path, info.toDrop + info.jobCodes + info.questionable)

    assert df.size #makes sure df actually exists

    print("This is the number of columns before dropping: " + str(df.columns.size))

    df = io.dropSameColumn(df)

    print("This is the number of columns after dropping: " + str(df.columns.size))

    df = io.oneHot(df)

    print("This is the number of columns after one hotting: " + str(df.columns.size))

    df.to_pickle('noQuestionable.pickle')

    print("All tests passed")
