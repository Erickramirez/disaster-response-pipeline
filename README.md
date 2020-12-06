# Disaster Response Pipeline Project
## Summary
This is proeject has a machine learning pipeline to build a model for an API that classifies disaster messages.
The model is used in a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the data.
### Prerequisites
The environment needed for this project:
1. [Python 3.6](https://www.python.org/downloads/release/python-360/)
2. [NumPy](https://numpy.org/)
3. [pandas](https://pandas.pydata.org/)
4. [nltk -Natural Language Toolkit](https://www.nltk.org/)
5. [SQLAlchemy](https://www.sqlalchemy.org/)
6. [scikit-learn](https://scikit-learn.org/stable/)
7. [pickle — Python object serialization](https://docs.python.org/3/library/pickle.html#module-pickle)
8. [Flask](https://flask.palletsprojects.com/en/1.1.x/)
9. [Plotly](https://plotly.com/python/)
10. [wordcloud](https://pypi.org/project/wordcloud/)

### Instructions to run the application
1. clone the github repository: `git clone https://github.com/Erickramirez/disaster-response-pipeline.git`
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://127.0.0.1:3001/ or as http://localhost:3001
### Explanation of the files in the repository
1. ETL Pipeline: folder: `/data` and contains the following files:
    - `process_data.py`, it performs data cleaning pipeline that:
        - Loads the messages and categories datasets
        - Merges the two datasets
        - Cleans the data, Note: the related category in order to have binary values has been created dimmies columns as related_1(0), related_2(1) and related_3(2) 
        - Stores it in a SQLite database
    - `disaster_categories.csv` categories datasets (data not cleaned)
    - `disaster_messages.csv` messages datasets (data not cleaned)
    - `DisasterResponse.db` output (dada cleaned) in a SQLite database
2. ML Pipeline: folder:`models` and contains the following files:
    - `train_classifier.py`, contains a machine learning pipeline that:
        - Loads data from the SQLite database (in the location `../data/DisasterResponse.db`)
        - Splits the dataset into training and test sets
        - Builds a text processing and machine learning pipeline
        - Trains and tunes a model using GridSearchCV
        - Outputs results on the test set
        - Exports the final model as a pickle file (classifier.pkl - not included in the repository due to its size)
3. Flask Web App: folder: `app`
    - `run.py` to run web application.
4. notebooks: .ipynb files for experimentation and they have been used as a base for "ETL Pipeline" and "ML Pipeline"
  
  

### How to use the web app
1. Overview of Training Dataset- review the following data description:
    - Categories Order by Count (in a bar)
    - Distribution of Message Genres (in a pie)
    - Most Frequent Words Used in a Disaster Message (in Word cloud)
2. Enter a message you want to label in the input field and click `Classify Message`. ![Web app view](/images/webapp.PNG)
### Conclusion
The classification report is in: `models/classification_report.txt` the data used in this report is:
1. `precision = (True positive) / (True positive + False positive)`
2. `recall = (True positive)/(True positive + False negative)`
3. `f1-score = 2*(precision*recall)/(precision+recall)`

The sample data average was: 0.74 as precision, 0.57 as recall and 0.59 as f1-score. Which looks that it is underfitting, a better aproach will be using a Recurrent neuronal network (RNN) like LSTM.
The machine of training was time-consuming, I tried to inprove the time using n_jobs = -1 to use all the cores at the same time.
