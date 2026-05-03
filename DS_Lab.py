import pandas as pd


def load_data(path):

    df = pd.read_csv(path)

    return df







def main():

    df = load_data("C:/Users/tomma/Downloads/Database_ASCII_EN/Database_ENG.csv")

    print(df.head())


if __name__ == "__main__":
    main()


