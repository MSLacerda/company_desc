import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.manifold import TSNE
import logging
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import MiniBatchKMeans
import string
import re


class KPipeline:

    wav2vec_model = None
    kmeans_model: KMeans = None 
    tsne: TSNE = None
    vectors: np.ndarray
    _df: pd.DataFrame

    def __init__(
            self,
    ):
        pass

    def preprocess_text(self, data: List[str]):
        """
        Preprocesses the input texts by removing stop words, lemmatizing tokens, and masking named entities using Spacy.

        Args:
            data (list): The list of input texts.

        Returns:
            list: The list of preprocessed texts.
        """
        nlp = spacy.load("en_core_web_md")
        docs = list(nlp.pipe(data, disable=["parser", "ner"], n_process=cpu_count()))

        preprocessed_texts = []
        for doc in tqdm(docs):
            preprocessed_tokens = []
            for token in doc:
                if not token.is_stop:
                    token_text = re.sub(r'[^\w\s]', '', token.text)
                    if token_text:
                        preprocessed_tokens.append(token.lemma_)
            
            preprocessed_text = " ".join(preprocessed_tokens)
            preprocessed_texts.append(preprocessed_text)

        return preprocessed_texts

    def vectorization(self, data: List[List[str]], wav2vec = None):
        """
        Converts preprocessed text data into numerical feature vectors using word2vec embeddings.

        Args:
            data (list): The list of preprocessed texts.

        Returns:
            numpy.ndarray: The numerical feature vectors.
        """

        if not self.wav2vec_model:
            self.wav2vec_model = wav2vec
        
        features = []

        for tokens in data:
            zero_vector = np.zeros(self.wav2vec_model.vector_size)
            vectors = []
            for token in tokens:
                if token in self.wav2vec_model.wv:
                    try:
                        vectors.append(self.wav2vec_model.wv[token])
                    except KeyError:
                        continue
            if vectors:
                vectors = np.asarray(vectors)
                avg_vec = vectors.mean(axis=0)
                features.append(avg_vec)
            else:
                features.append(zero_vector)

        return features
        
    def prepare(
            self,
            df: pd.DataFrame,
            text_column: str,
            w2v_model
        ):
            self._df = df.copy()

            # Extract the texts from the specified column
            texts =  self._df[text_column].tolist()

            # Preprocess the texts
            logging.info('Preprocessing texts...')
            preprocessed_texts = self.preprocess_text(texts)

            # Convert the preprocessed texts into numerical feature vectors the wav2vec
            logging.info('Vectorizing the texts...')
            self.vectors = self.vectorization(preprocessed_texts, w2v_model)

            return self.vectors
    

    def mbkmeans_clusters(
        self,
        k: int, 
        mb: int, 
        print_silhouette_values: bool = False, 
    ):
        """Generate clusters and print Silhouette metrics using MBKmeans

        Args:
            k: Number of clusters.
            mb: Size of mini-batches.
            print_silhouette_values: Print silhouette values per cluster.

        Returns:
            Trained clustering model and labels based on X.
        """

        if len(self.vectors) == 0:
            logging.error('Please run pipeline.prepare, before training')

            raise ValueError

        km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(self.vectors)
        print(f"For n_clusters = {k}")
        print(f"Silhouette coefficient: {silhouette_score(self.vectors, km.labels_):0.2f}")
        print(f"Inertia:{km.inertia_}")

        if print_silhouette_values:
            sample_silhouette_values = silhouette_samples(self.vectors, km.labels_)
            print(f"Silhouette values:")
            silhouette_values = []
            for i in range(k):
                cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
                silhouette_values.append(
                    (
                        i,
                        cluster_silhouette_values.shape[0],
                        cluster_silhouette_values.mean(),
                        cluster_silhouette_values.min(),
                        cluster_silhouette_values.max(),
                    )
                )
            silhouette_values = sorted(
                silhouette_values, key=lambda tup: tup[2], reverse=True
            )
            for s in silhouette_values:
                print(
                    f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
                )

        self.kmeans_model = km

        return self.kmeans_model
    
    def predict_companies(self, user_requirements: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict and recommend a set of companies based on user requirements.
        
        Args:
            user_requirements (str): User's specific requirements or preferences.
            df (pd.DataFrame): DataFrame containing company descriptions.
            vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
            pca (PCA): Trained PCA for dimensionality reduction.
            
        Returns:
            pd.DataFrame: DataFrame with recommended companies matching user requirements.
        """
        # Preprocess user requirements
        preprocessed_requirements = self.preprocess_text([user_requirements])

        # Vectorize user requirements
        vectors = self.vectorization(preprocessed_requirements)

        # Apply PCA and tSNE for dimensionality reduction
        # data_tsne = self.tsne.fit_transform(vectors)

        # Predict cluster for user requirements
        cluster_label = self.kmeans_model.predict(vectors)[0]

        # Get companies belonging to the predicted cluster
        recommended_companies = df[df['cluster_label'] == cluster_label]

        return recommended_companies