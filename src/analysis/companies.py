import re
import spacy
import pandas as pd
from tqdm import tqdm

def detect_company_name(df: pd.DataFrame, text_column: str, nlp_model: spacy.language.Language) -> pd.DataFrame:
    """
    Detects company names from the given dataframe's text column using spaCy's entity type, PoS, and dependency information.
    
    Args:
        df (pd.DataFrame): The input dataframe containing the text column.
        text_column (str): The name of the column containing the company descriptions.
        nlp_model (spacy.lang.en.Language): The spaCy language model to use for entity detection.
        
    Returns:
        pd.DataFrame: The input dataframe with a new column 'name' containing the detected company names.
    """

    _df = df.copy()

    company_names = []

    for text in tqdm(_df[text_column].tolist()):
        company_name = ''

        doc = nlp_model(text)

        for sent in doc.sents:
            for token in sent:
                if (token.ent_type_ == 'ORG' or token.ent_type_ == 'PRODUCT') and \
                  (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass') and \
                  (token.pos_ == 'PROPN'):
                    company_name = token.text

                    if token.left_edge.dep_ == 'compound':
                        company_name = f'{token.left_edge.text} {company_name}'
                    
                    if token.right_edge.dep_ == 'compound':
                        company_name = f'{company_name} {token.left_edge.text}'
                    
                    break

            if company_name:
                break

        if not company_name:
            # Fallback solution if the algorithm above doesnt work
            # this is will get the fist word or two words with capital letter
            pattern = r'\b[A-Z0-9+][a-zA-Z]+ ?([A-Z][a-zA-Z]+)?\b'
            matched = re.search(pattern, text)
            
            if matched:
                company_name = matched.group()
                
        
        if company_name.lower().startswith('at '):
            company_name = company_name.lstrip('at ')

        company_names.append(company_name)

    _df['name'] = company_names

    return _df
