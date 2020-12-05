import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load csv files (2 sources) and merge them into one dataframe

    :param messages_filepath:  file location of csv file with messages
    :param categories_filepath: file location of csv file with categories
    :return: dataframe with merge data of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id', how='left')


def clean_data(df):
    """
    clean data, create new columns based on categories column and remove not useful data.
    :param df: dataframe to clean the data
    :return: data fram with columns based on categories
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.loc[0]

    categories.columns = row.str.split('-').str.get(0)
    for column in categories: #create new columns based on the distinct values in the column category
        categories[column] = categories[column].str.get(-1)
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1) # concatenate the two dataframes
    df = df.drop_duplicates()
    df.related = df.related.replace(2, 1) #related has 3 posible values: 0,1,2. use only 0 and 1 instead
    df = df.drop(['child_alone'], axis=1) #remove useless column
    return df


def save_data(df, database_filename):
    """
    save dataframe into a SQL lite DB file

    :param df: dataframe to save
    :param database_filename: file location (extension .db)
    :return: no return
    """
    connection = 'sqlite:///{}'.format(database_filename)
    engine = create_engine(connection)
    df.to_sql('disaster_message', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()