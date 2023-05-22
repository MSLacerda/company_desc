import re
import pandas as pd
from typing import List
from spacy.language import Language
from tqdm import tqdm

def normalize_text(df: pd.DataFrame, columns: List[str] = []) -> pd.DataFrame:
    """
    Normalizes the text in the specified columns of a pandas DataFrame by removing non-printable characters, line breaks,
    duplicated spaces, and leading/trailing spaces.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (List[str]): The list of columns to normalize.
        
    Returns:
        pandas.DataFrame: The DataFrame with normalized text in the specified columns.
    """

    _df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original data

    # Apply text normalization operations to each cell in the specified columns
    _df[columns] = _df[columns].applymap(lambda x: re.sub(r'[^ -~]', '', str(x)))
    _df[columns] = _df[columns].applymap(lambda x: re.sub('\r|\n', ' ', str(x)))
    _df[columns] = _df[columns].applymap(lambda x: re.sub(' +', ' ', str(x)).strip())

    return _df

def remove_noisy_texts(
    df: pd.DataFrame, 
    columns: List[str], 
    patterns: List[str],
    show_count: bool = False
) -> pd.DataFrame:
    """
    Removes specified patterns from the specified columns of a pandas DataFrame along with any special characters that follow them.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): The list of columns to remove noisy texts from.
        show_count (bool): Whether to display the count of noisy patterns found.
        
    Returns:
        pd.DataFrame: The DataFrame with noisy texts and special characters removed from the specified columns.
    """
    _df = df.copy()
    count = 0

    if show_count:
        count = _df[columns].apply(lambda x: x.str.contains('|'.join(patterns), flags=re.IGNORECASE).sum(), axis=0)
        print("Noisy pattern counts:")
        print(count)

    for column in columns:
        for pattern in patterns:
            regex_pattern = r'{}\W*'.format(re.escape(pattern))
            _df[column] = _df[column].apply(lambda x: re.sub(regex_pattern, '', str(x), flags=re.IGNORECASE))

        _df[column] = _df[column].str.strip()
    
    return _df

def add_textual_statistics(df: pd.DataFrame, text_column: str, spacy_model: Language) -> pd.DataFrame:
    """
    Augments a DataFrame with additional textual statistics derived from the specified column using spaCy.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_column (str): The name of the textual column in the DataFrame.
        
    Returns:
        pandas.DataFrame: The augmented DataFrame with additional textual features.
    """

    _df = df.copy()
    
    # Create empty lists to store the features
    text_lengths = []
    num_sentences = []
    token_counts = []
    
    for _, row in tqdm(_df.iterrows(), total=len(_df.index)):
        text = row[text_column]
        
        # Process the text using spaCy
        doc = spacy_model(text)
        
        # Calculate the length of the text in characters
        text_lengths.append(len(text))
        
        # Count the number of sentences in the text
        num_sentences.append(len(list(doc.sents)))
        
        # Count the number of tokens in the text
        token_counts.append(len(doc))
    
    # Add the new features to the DataFrame
    _df['text_length'] = text_lengths
    _df['num_sentences'] = num_sentences
    _df['token_count'] = token_counts
    
    return _df