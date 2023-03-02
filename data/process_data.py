# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    returns a merged dataframe given two different datasets
    
    INPUT:
        messages_filepath: path to read messages csv file
        categories_filepath: path to read categories csv file
    
    OUTPUT:
        merged dataframe based on "id" column
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    
    '''
    returns a cleaned dataframe where the next steps are followed: 
    
    1. Creation of a dataframe with 36 columns of all the different categories
    2. Obtention of the categories of the dataset and using them as new column names
    3. Change the values from text to binary numbers in each of the new 36 columns
    4. Duplicate dropping
    
    INPUT: 
        df: dataframe to be cleaned
    
    OUTPUT:
        cleaned dataframe
    '''
    
    
    # Creation of a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # First row of the categories dataframe to get the different values
    row = pd.Series(df.categories[0])

    # Separation of category names with split and apply functions
    category_colnames = row.str.split(pat=';', expand=True).apply(lambda value: value.str.slice(0,-2))

    # Dataframe to list
    category_colnames = category_colnames.values.tolist()

    # Flatten of list
    category_colnames = [item for elem in category_colnames for item in elem]
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    
    # Loop to change columns to numerical values
    for column in categories:
    
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # Convertion of column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Dropping the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop columns that have complete zeros in all column
    df = df.loc[:, (df != 0).any(axis=0)]
    
    # Drop rows that have a 2 on related column
    df = df[df['related'] != 2]
    
    return df
        

def save_data(df, database_filename):
    
    '''
    save dataframe to a sqlite database
 
    INPUT: 
        df: dataframe to be saved
        database_filename: name of the database where the dataframe will be stored
    
    OUTPUT:
        cleaned dataframe
    '''
    
    engine = create_engine(database_filename)
    df.to_sql('messages_categories', engine, index=False) 


def main():
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        database_filepath = 'sqlite:///' + database_filepath 
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()