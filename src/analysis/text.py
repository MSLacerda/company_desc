import pandas as pd
import numpy as np
from typing import List, Tuple
from collections import Counter
from spacy import Language
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def classify_outliers(df: pd.DataFrame, columns: List[str],):
    """
    Obtains the outliers classification based on the specified column in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the column to analyze for outliers.

    Returns:
        pandas.DataFrame: A DataFrame containing the outliers columns classification.
    """


    _df = df.copy()

    for column in columns:
        # Calculate the bounds for outliers
        q1 = _df[column].quantile(0.25)
        q3 = _df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers and add a column to indicate upper or lower bound
        _df[f'{column}_outlier_bound'] = np.where((_df[column] < lower_bound) | (_df[column] > upper_bound),
                                    np.where(_df[column] < lower_bound, 'lower_bound', 'upper_bound'), 'no_outlier')

    return _df


def calculate_most_frequent_tokens(data: List[str], model: Language) -> List[Tuple[str, int]]:
    """
    Calculates the most frequent tokens in the specified column of a dataset using spaCy.

    Args:
        data (List[str]): The list of text data.
        column (str): The name of the column containing the textual data.
        model (spacy.Language): The spaCy language model.

    Returns:
        List[Tuple[str, int]]: A list of tuples, where each tuple contains the token and its frequency.
    """
    # Join all the text data in the specified column into a single string
    text = " ".join(data)

    # Process the text using spaCy
    doc = model(text)

    # Extract the tokens and their frequencies
    tokens = [token.text for token in tqdm(doc, total=len(doc)) if not token.is_stop and not token.is_punct]
    token_freqs = Counter(tokens)

    # Sort the tokens by frequency in descending order
    sorted_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)

    return sorted_tokens



def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """

    sns.set_style("darkgrid")
    arrays = np.empty((0, 50), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=min(arrays.shape[0], arrays.shape[1])).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))