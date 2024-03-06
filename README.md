# Wiki_Infromation_retreval
Wiki search engine

# About
This is a final project for the cource *Information retrival*.
We are 3 student from BGU who with the help of our professor *Nir Grinberg* created a search engine on the wikipedia in the language of english.
This rep also contains all of the code to create it by urself with the use of GCP (look at instruction).

# Requirements
1. Installation of google cloud storage
```basch
pip install -q google-cloud-storage==1.43.0
```
2. You will have to create uself a project on Goole Cloud console, and follow the *instruction* file under Index creations.
  
3. Python 3.6 or later
 ```basch
py -m pip install [Package_to_install]
```
4. Flask

5. NLTK

6. gzip

7. csv

8. numpy
 ```basch
pip install numpy
```
9. pandas
 ```basch
pip install pandas
```

# Usage
* Run the command python3 search_frontend.py to start the application
* The application will be running on http://External_IP:8080/ by default
* To issue a query, navigate to a URL like http://External_IP:8080/search?query=hello+world

# Data
* The corpus of documents is a collection of Wikipedia articles
* The inverted index , PageRank , Pageview data is pre-computed and included in the project files
* Additional data, such as document lengths,term frequency  is calculated before the application starts

