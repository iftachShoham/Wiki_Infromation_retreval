import csv
import gzip
import io
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage
import math
import threading

import hashlib


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')

from inverted_index_gcp import *

class ThreadSafeCounter(Counter):
  def _init_(self):
    super()._init_()
    self.lock = threading.Lock()

  def _setitem_(self, key, value):
    with self.lock:
      super()._setitem_(key, value)

class ThreadSafeCounter(Counter):
  def __init__(self):
    super().__init__()
    self.lock = threading.Lock()

  def __setitem__(self, key, value):
    with self.lock:
      super().__setitem__(key, value)

from LLMImprove import *

class BackEnd():
  """
  Class that handle all of the backEnd of the project.
  """
  def __init__(self):

    # Number of docs in the corpus
    self.N = 6348910

    # For bm25:
    self.k1 = 1.0
    self.b = 0.9
    self.k2 = 10.0

    # For inverted index of the text alone:
    self.bucket_name_text = "bucket_ir_102"
    self.base_dir_inverted = "postings_gcp"
    self.text_InvertedIndex = InvertedIndex.read_index(self.base_dir_inverted, "index", self.bucket_name_text)

    # For inverted index of the title alone:
    self.bucket_name_title = "bucket_ir_101"
    self.title_InvertedIndex = InvertedIndex.read_index(self.base_dir_inverted, "index_title", self.bucket_name_title)

    # For Page rank dict:
    bucket_name_for_page_rank = "bucket_ir_100"
    file_name_for_page_rank = "pr/part-00000-bba051bd-4ed5-42d2-ac51-b81e7da0af95-c000.csv.gz"
    self.page_rank_dict = self.read_csv_gzip_to_dict(bucket_name_for_page_rank, file_name_for_page_rank)

    # For title_id:
    bucket_name_for_title_id = "bucket_ir_100"
    file_name_for_title_id = "id_title/part-00000-037780c9-c08d-4f8b-92db-c059071f2db8-c000.csv.gz"
    self.title_id = self.read_csv_gzip_to_dict_with_string(bucket_name_for_title_id, file_name_for_title_id)

    # For doc_legnth dict:
    bucket_name_for_doc_len = "bucket_ir_100"
    file_name_for_doc_len = "word_counts/part-00000-1487dca0-a7d2-4dd0-b547-440beeb5f720-c000.csv.gz"
    self.doc_length_dict = self.read_csv_gzip_to_dict(bucket_name_for_doc_len, file_name_for_doc_len)

    self.avg_doc_length = 0
    for length in self.doc_length_dict.values():
      self.avg_doc_length += length

    self.avg_doc_length = self.avg_doc_length / len(self.doc_length_dict)  # Assuming self.N is the number of documents


    # For stemming
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

    self.all_stopwords = english_stopwords.union(corpus_stopwords)
    self.RE_WORD = re.compile(r"\b\w{2,24}\b", re.UNICODE)

    self.question_words = {
                        'who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'whose',
                        'are', 'is', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
                        'may', 'might', 'shall', 'must', 'have', 'has', 'had', 'am', 'was', 'were',
                        'had', 'have', 'having', 'been', 'being', 'if', 'whether', 'whenever', 'wherever',
                        'whatever', 'whichever', 'whoever', 'whomever', 'whyever', '?'}


  def read_csv_gzip_to_dict(self, bucket_name, file_name):
    """
    Reads csv from the bucket, given the file name into a dict and returns it.
    key = doc_id
    value = page rank
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the file as bytes
    content = blob.download_as_string()

    # Decompress the gzip file and read it as a CSV
    with gzip.open(io.BytesIO(content), "rt") as gzip_file:
        csv_reader = csv.reader(gzip_file)
        header = next(csv_reader)  # Assuming the first row is the header
        data = {int(row[0]): float(row[1]) for row in csv_reader}

    return data

  def read_csv_gzip_to_dict_with_string(self, bucket_name, file_name):
    """
    Reads csv from the bucket, given the file name into a dict and returns it.
    key = doc_id
    value = title
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the file as bytes
    content = blob.download_as_string()

    # Decompress the gzip file and read it as a CSV
    with gzip.open(io.BytesIO(content), "rt") as gzip_file:
      csv_reader = csv.reader(gzip_file)
      header = next(csv_reader)  # Assuming the first row is the header
      data = {}
      for row in csv_reader:
        try:
          doc_id = int(row[1])
        except ValueError:
          # Skip rows where doc_id is not a valid integer
          continue
        data[doc_id] = row[0]

    return data

  def bm25(self, query, tf_idf_dict):
    # create an empty Counter object to store document scores
    candidates = Counter()

    # loop through each term in the query
    for word in query:
      # check if the word exists in the corpus
      if word in self.text_InvertedIndex.df:
        # read the posting list of the word
        posting_list = self.text_InvertedIndex.read_a_posting_list(self.base_dir_inverted, word, self.bucket_name_text)

        # calculate idf of the word
        df = self.text_InvertedIndex.df[word]
        idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

        # loop through each (doc_id, freq) pair in the posting list
        for doc_id, freq in posting_list:

          len_doc = self.doc_length_dict.get(doc_id, -1)
          if len_doc == -1:
            continue

          # calculate bm25 score of the word for the document
          numerator = idf * freq * (self.k1 + 1)
          denominator = (freq + self.k1 * (1 - self.b + self.b * len_doc / self.avg_doc_length))
          bm25_score = numerator / denominator
          bm25_score = bm25_score * ((self.k2 + 1) * freq / (self.k2 + freq))

          # add the bm25 score to the document's score in the candidates Counter
          candidates[doc_id] += bm25_score
    tf_idf_dict.update(candidates)

  # def tf_idf(self, query, tf_idf_dict):
  #   """
  #   args:
  #   query - stemmed and trimmed list of the words that were searched.
  #   tf_idf_dict- counter
  #
  #   returns a counter:
  #     key = doc_id
  #     value = W (int) calculated by the tf-idf
  #     with the words asked.
  #   """
  #
  #   for word in query:
  #       # Only handle words that are in the dictionary.
  #       if word in self.text_InvertedIndex.df:
  #           term_df = self.text_InvertedIndex.df[word]
  #           pl = self.text_InvertedIndex.read_a_posting_list(self.base_dir_inverted, word, self.bucket_name_text)
  #           # Map it to get -> [(doc_id,W)....]
  #           pl_map = map(lambda x: (x[0],
  #                       (x[1]/self.doc_length_dict.get(x[0],1))*(math.log2(self.N/term_df))), pl)
  #
  #           # Update tf_idf_dict with the new values, summing if the key already exists
  #           tf_idf_dict.update(dict(pl_map))
  #
  #   return tf_idf_dict

  def tf_idf(self, query, tf_idf_dict):
    """
  args:
  query - stemmed and trimmed list of the words that were searched.
  BM25_dict- counter

  returns a counter:
    key = doc_id
    value = W (int) calculated by the tf-idf
    with the words asked.
  """
    # Create a defaultdict to store the cosine similarity of each document to the query
    similarities = defaultdict(int)
    # Putting back the query to sentence and create a counter of the query terms
    counter_query = Counter(query)
    for word in query:
      if word in self.text_InvertedIndex.df:
        term_df = self.text_InvertedIndex.df[word]
        # Calculate the inverse document frequency (IDF) for the term
        idf = math.log2(self.N / term_df)
        # Read the posting list for the term
        pl = self.text_InvertedIndex.read_a_posting_list(self.base_dir_inverted, word, self.bucket_name_text)
        # For each document where the term appears, calculate the term frequency (TF)
        # and weight it by the IDF
        for doc_id, freq in pl:
          tf = freq / self.doc_length_dict.get(doc_id, 2000)
          weight = tf * idf
          # Add the weighted term to the document's similarity score
          similarities[doc_id] += weight * counter_query[word]

    # Normalize the similarity score by the query and document lengths
    normalization_query = 0
    sum_q = 0

    # Calculate the length of the query vector
    for term, freq in counter_query.items():
      sum_q += freq * freq
    normalization_query = 1 / math.sqrt(sum_q)

    # For each document, normalize the similarity score by the document length and the query length
    for doc_id in similarities.keys():
      nf = 1 / math.sqrt(self.doc_length_dict.get(doc_id, 2000))
      similarities[doc_id] *= normalization_query * nf

    # Update the dict
    tf_idf_dict.update(similarities)


  def page_rank_refactoring(self, tf_idf_dict, factor_value_pr=1):
    """
    After the tf-idf has been done, it will refactor the result
    with Page rank.
    """
    for k in tf_idf_dict.keys():
      tf_idf_dict[k] += math.log2(factor_value_pr * self.page_rank_dict.get(k, 1))



  def title_refactoring(self, query, tf_idf_dict, factor_value_title=1):
      """
      After the tf-idf has been done, it will refactor the result
      with title factoring, assuming if the words were in the title it
      should have more weight.
      """
      for word in query:
      # Only handle words that are in the dictionary.
        if word in self.title_InvertedIndex.df:
          term_df = self.title_InvertedIndex.df[word]
          pl = self.title_InvertedIndex.read_a_posting_list(self.base_dir_inverted, word, self.bucket_name_title)
          # Map it to get -> [(doc_id,W)....]
          pl_map = map(lambda x: (x[0], factor_value_title * (1 /max(1,len(self.title_id.get(x[0], [])))) * max((math.log2(1 + (self.N / term_df))), 1)), pl)
          # Update tf_idf_dict with the new values, summing if the key already exists
          tf_idf_dict.update(dict(pl_map))


  def filter_query_with_stemming(self, query):
    """
    This will recive a raw string of the query and stemm it,
    remove stopping words and creating a list of each token in it.
    returning the list.
    """
    stemmer = PorterStemmer()
    tokens = re.findall(self.RE_WORD, query.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    filtered_tokens = [token for token in stemmed_tokens if token not in self.all_stopwords]
    return filtered_tokens

  def filter_query_title(self, query):
    stop_words = set(stopwords.words('english'))
    word_tokens = re.findall(self.RE_WORD, query.lower())
    filtered_query = [word for word in word_tokens if word.lower() not in self.all_stopwords]
    return filtered_query

  def filter_query_without_stemming(self, query):
    """
    This will recive a raw string of the query and stemm it,
    remove stopping words and creating a list of each token in it.
    returning the list.
    """
    tokens = re.findall(self.RE_WORD, query.lower())
    filtered_tokens = [token for token in tokens if token not in self.all_stopwords]
    return filtered_tokens


  def question_wiki(self, query, use_title=True,use_page_rank=True,TITLE_VALUE_FACTORING=1,PAGE_RANK_FACTORING=1,use_multy_thread=True):
    """
    Going through the entire process, given a string of the question asked,
    it will return the 100 most fitted wiki_id_pages and its title.
    """
    # Checking special condition
    # if '"' in query:
    #   return self.question_wiki_with_par(query,use_title=use_title,use_page_rank=use_page_rank,TITLE_VALUE_FACTORING=TITLE_VALUE_FACTORING,PAGE_RANK_FACTORING=PAGE_RANK_FACTORING)

    # Using llm to improve the query

    llm_q = LLMImprove.improve_query(query)
    # # Step 1: correct typos.
    # query = self.correct_typos(query)
    original_q = self.filter_query_title(llm_q)

    if "when" in query or "When" in query:
      search_method = self.bm25
    else:
      search_method = self.tf_idf
      query = llm_q


    # flag_for_question = True
    # for word in original_q:
    #   if word in self.question_words:
    #     PAGE_RANK_FACTORING = PAGE_RANK_FACTORING * 250
    #     flag_for_question = False
    #     break

    # # Use regular expression to find individual words within double quotes
    # words_within_quotes = re.findall(r'"([^"]*)"', query)
    # # Join the words into a sentence
    # sentence_within_quotes = ' '.join(words_within_quotes)
    # # query = self.correct_typos(query)
    # list_of_words_within_quotes = self.filter_query_with_stemming(sentence_within_quotes)


    # Step 2: filter and stem and if the query length is less than 3, so also add the unstemmed words.
    if len(query) < 3:
        query = self.filter_query_without_stemming(query) + self.filter_query_with_stemming(query)
        TITLE_VALUE_FACTORING = TITLE_VALUE_FACTORING * 2
    else:
        query = self.filter_query_with_stemming(query)

    # step 3: check if their question words in the query


    # Step 4: tf_idf on the body
    # Now its multy threaded.
    threads = []
    tf_idf_dict = ThreadSafeCounter()

    if use_multy_thread:
      for q in query:
          thread = threading.Thread(target=search_method, args=([q], tf_idf_dict))
          threads.append(thread)
          thread.start()

    # Step 4: tf_idf on the body not multy threaded
    else:
      search_method(query, tf_idf_dict)


    # Step 5: use the title factoring
    if use_title and not use_multy_thread:
      self.title_refactoring(original_q, tf_idf_dict, TITLE_VALUE_FACTORING)

    # Use the title factoring not multy threaded
    if use_title and use_multy_thread:
      for q in original_q:
          thread = threading.Thread(target=self.title_refactoring, args=([q], tf_idf_dict, TITLE_VALUE_FACTORING))
          threads.append(thread)
          thread.start()

    # Make sure to join all of the threads before the pagerank.
    if use_multy_thread:
      for thread in threads:
        thread.join()

    # Checking to see if the dict is too small, if so ill lower the stemming
    if len(tf_idf_dict) < 100:
      threads = []
      if use_multy_thread:
        for q in original_q:
          thread = threading.Thread(target=search_method, args=([q], tf_idf_dict))
          threads.append(thread)
          thread.start()
        for thread in threads:
          thread.join()

      # Step 3: tf_idf on the body not multy threaded
      else:
        search_method(original_q, tf_idf_dict)


    # Step 5: use the page rank refactoring
    if use_page_rank:
      self.page_rank_refactoring(tf_idf_dict, PAGE_RANK_FACTORING)

    # Step 6: Return top 100 results.

    # Convert the Counter to a list of tuples and sort it by value in descending order
    sorted_counter = sorted(tf_idf_dict.items(), key=lambda x: x[1], reverse=True)
    # Get the top 100 items
    top_100_keys = [
      (str(doc_id), self.title_id[doc_id])
      for doc_id, value in sorted_counter[:min(100, len(sorted_counter))]
      if doc_id in self.title_id
    ]

    return top_100_keys

  # def question_wiki_with_par(self, query, use_title=True,use_page_rank=True,TITLE_VALUE_FACTORING=1,PAGE_RANK_FACTORING=1,use_multy_thread=True):
  #   # Getting the words insied the ""
  #   pattern = r'"(.*?)"'
  #   query = re.findall(pattern, query)

  #   query = LLMImprove.improve_query(" ".join(query))

  #   # Steming
  #   query = self.filter_query_with_stemming(query)

  #   similarities = defaultdict(int)
  #   # Putting back the query to sentence and create a counter of the query terms
  #   counter_query = Counter(query)
  #   flag_first = True

  #   for word in query:
  #     if word in self.text_InvertedIndex.df:
  #       term_df = self.text_InvertedIndex.df[word]
  #       # Calculate the inverse document frequency (IDF) for the term
  #       idf = math.log2(self.N / term_df)
  #       # Read the posting list for the term
  #       pl = self.text_InvertedIndex.read_a_posting_list(self.base_dir_inverted, word, self.bucket_name_text)
  #       for doc_id, freq in pl:
  #         # Making sure the words already exist
  #         if doc_id not in similarities or flag_first:
  #           tf = freq / self.doc_length_dict.get(doc_id, 2000)
  #           weight = tf * idf
  #           # Add the weighted term to the document's similarity score
  #           similarities[doc_id] += weight * counter_query[word]

  #       flag_first = False
  #   # Normalize the similarity score by the query and document lengths
  #   normalization_query = 0
  #   sum_q = 0

  #   # Calculate the length of the query vector
  #   for term, freq in counter_query.items():
  #     sum_q += freq * freq
  #   normalization_query = 1 / math.sqrt(sum_q)

  #   # For each document, normalize the similarity score by the document length and the query length
  #   for doc_id in similarities.keys():
  #     nf = 1 / math.sqrt(self.doc_length_dict.get(doc_id, 2000))
  #     similarities[doc_id] *= normalization_query * nf


  #   if use_page_rank:
  #     self.page_rank_refactoring(similarities, PAGE_RANK_FACTORING)

  #   # Convert the Counter to a list of tuples and sort it by value in descending order
  #   sorted_counter = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
  #   # Get the top 100 items
  #   top_100_keys = [
  #     (str(doc_id), self.title_id[doc_id])
  #     for doc_id, value in sorted_counter[:min(100, len(sorted_counter))]
  #     if doc_id in self.title_id
  #   ]
  #   return top_100_keys



  def get_posting_list_body(self, word):
    """
    return the posting list of a word by the text index
    """
    return self.text_InvertedIndex.read_a_posting_list(self.base_dir_inverted, word, self.bucket_name_text)

  def get_posting_list_title(self, word):
    """
    return the posting list of a word by the title index
    """
    return self.title_InvertedIndex.read_a_posting_list(self.base_dir_inverted, word, self.bucket_name_title)

  def get_page_rank(self, doc_id):
    """
    returns the page rank by the doc id.
    """
    return self.page_rank_dict.get(doc_id, 0)

  def get_doc_length(self, doc_id):
    # Returns the doc_length by the doc id.
    return self.doc_length_dict.get(doc_id,1)

class ThreadSafeCounter(Counter):
  def __init__(self):
    super().__init__()
    self.lock = threading.Lock()

  def __setitem__(self, key, value):
    with self.lock:
      super().__setitem__(key, value)




