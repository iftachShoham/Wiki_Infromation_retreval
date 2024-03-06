# Wiki_Infromation_retreval

This project was completed as part of a university Information Retrieval course, aiming to develop a search engine for retrieving relevant information from a corpus of Wikipedia articles. The data retrieval process involved creating an inverted index on Google Cloud Platform (GCP), indexing the body, title, and anchor text of the Wikipedia articles. The search functionality was implemented in Python using the Flask framework, leveraging the inverted index for result retrieval. To enhance the search engine's efficiency, we pre-computed certain values such as document lengths and stored them in dictionaries to avoid runtime calculations. Additionally, we employed the PageRank algorithm for result ranking and used techniques like stopword removal to enhance retrieval effectiveness.

# About
This is a final project for the course *Information retrival*.
We are 3 student from BGU who with the help of our professor *Nir Grinberg* created a search engine on the wikipedia in the language of english.
This rep also contains all of the code to create it by urself with the use of GCP (look at instruction).

# Requirements
1. Installation of google cloud storage
```basch
pip install -q google-cloud-storage==1.43.0
```  
2. Python 3.6 or later
 ```basch
py -m pip install [Package_to_install]
```
* Flask
* NLTK
* gzip
* csv
* numpy
* pandas

# DIY (Do It yourself)
If you wish to create it by usrself follow the following steps:

1. Follow the *instruction* file, and create urself a Google Cloud Console.
   
2. Use the code to create urself all of the indexes.
   
3.  run the front_end and queriy urself!

**Tip:** Use the test to test each of your index in order to be sure it works!

# Usage
* Run the command python3 search_frontend.py to start the application
* The application will be running on http://External_IP:8080/ by default
* To issue a query, navigate to a URL like http://External_IP:8080/search?query=hello+world

# Data
* The corpus of documents is a collection of Wikipedia articles
* The inverted index , PageRank , Pageview data is pre-computed and included in the project files
* Additional data, such as document lengths,term frequency  is calculated before the application starts

