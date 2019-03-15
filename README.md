# Exploratory Analysis of Hotel Guest Ratings Data using Machine Learning Algorithms
## Introduction
What compels people to take the time to write a review about the hotels they stayed at? What factors influence their ratings? These are the kinds of information property owners and managers can glean insights on to improve services and facilities, to determine the sentiments of their consumer niche, and to perform at par or better than their competition.

In this project, data on consumer reviews and other data of over 1,800 hotels was used to develop machine learning algorithms to find factors that lead to hotel guest ratings.

### Objectives
1. To create a model that can be used to predict hotel ratings given by visitors based on various hotel features.
2. To obtain sentiment insights from the visitors who provided information about their experiences in the various properties.

## Libraries used
### Data processing
1. [pandas](https://pandas.pydata.org/): for creating dataframes and other data types 
2. [numpy](http://www.numpy.org/): creation of arrays and scientific computing
3. [pyspark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html): allows Spark programming functionality in the Python environment
4. [collections](https://docs.python.org/2/library/collections.html): specialised data type alternatives to dictionaries, lists, sets, and tuples
5. [sqlite3](https://docs.python.org/3.4/library/sqlite3.html): allows SQLite functionality in the Python environment
### Machine learning
1. [sklearn](https://scikit-learn.org/stable/): machine learning module in Python with functionalities for regression, classification, dimension reduction, etc.
2. [mord](https://pythonhosted.org/mord/): machine learning module with sklearn functionality that is used for ordinal regression
3. [statsmodels](https://www.statsmodels.org/stable/index.html): statistical tests and data exploration, including linear regression
4. [nltk](https://www.nltk.org/): library that is used to process text data (natural language processing)
### Visualisation
1. [matplotlib](https://matplotlib.org/): library for creating graphs and charts using Python
2. [seaborn](https://seaborn.pydata.org/): library based on matplotlib; for creating graphs and charts
3. [wordcloud](https://github.com/amueller/word_cloud): word cloud generator in Python

## Jupyter notebooks
The following Jupyter notebooks were used to process the raw data, to conduct machine learning, and to visualise the results.
<table>
    <tr>
        <th>File name</th> <th>Description</th>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis.ipynb">Hotel_Ratings_Analysis.ipynb</a></td>
        <td>
            <ul>
                <li>Extraction of data from <a href = "https://data.world">Data World</a> on hotel reviews</li>
                <li>Created a SQLite database that has table for <i>metadata</i> and for <i>ratings</i></td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-02.ipynb">Hotel_Ratings_Analysis-02.ipynb</a></td>
        <td>Further cleaning of hotel metadata
            <ul>
                <li>Extraction of <a href = "https://opendata.socrata.com/dataset/Airport-Codes-mapped-to-Latitude-Longitude-in-the-/rxrh-4cxm">geographical coordinates</a> of more than 13,000 airports in the USA</li>
                <li>Calculation of distances between each hotel and the nearest airport to it (in km)</li>
                <li>Tokenisation of hotel categories and development of dummy variables for each hotel characteristic</li>
                <li>Development of dummy variables for hotel provinces</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-03.ipynb">Hotel_Ratings_Analysis-03.ipynb</a></td>
        <td>Calculations of ratings from hotel reviewers
            <ul>
                <li>Average ratings by hotel booking website (e.g., tripadvisor, citysearch)</li>
                <li>Average ratings by hotel and by hotel booking website</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-04.ipynb">Hotel_Ratings_Analysis-04.ipynb</a></td>
        <td>Natural language processing (NLP) on each hotel review text
            <ul>
                <li>Word frequencies and heatmap</li>
                <li>Word clouds</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-05.ipynb">Hotel_Ratings_Analysis-05.ipynb</a></td>
        <td>Combined ratings and metadata
            <ul>
                <li>Merge based on "Name"</li>
                <li>Calculated correlation coefficients (phi coefficient)</li>
                <li>Removed one variable from each highly correlated variable pair</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-06.ipynb">Hotel_Ratings_Analysis-06.ipynb</a></td>
        <td>Natural language processing (NLP) on each hotel review title
            <ul>
                <li>Word frequencies and heatmap</li>
                <li>Word clouds</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-07.ipynb">Hotel_Ratings_Analysis-07.ipynb</a></td>
        <td>Supervised learning using regression analyses
            <ul>
                <li>Removed zero-variance variables</li>
                <li>Random forest to find top 10 important variables</li>
                <li>Created five regression models and evaluated them</li>
                <li>Conducted PCA to <i>try</i> reducing the dimensionality</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-08.ipynb">Hotel_Ratings_Analysis-08.ipynb</a></td>
        <td>Determination of important words using TF-IDF
            <ul>
                <li>Combined titles and text</li>
                <li>Calculated TF-IDF</li>
                <li>Visualised important words using interactive seaborn horizontal bar graphs</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Hotel_Ratings_Analysis-09.ipynb">Hotel_Ratings_Analysis-09.ipynb</a></td>
        <td>Relationship between state population and number of hotel reviews determined using linear regression</td>
    </tr>
</table>

## Database
The data extracted and transformed in Jupyter notebooks were saved in <a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Data/Hotels.db">Hotels.db</a>. It contains three tables:
1. <b>metadata.</b> Raw data.
2. <b>metadata2.</b> Dummy variables and nearest-airport distance indicated.
3. <b>ratings.</b> Scores and reviews obtained from hotel visitors.
4. <b>alldata.</b> Merged data from metadata2 and ratings; for machine learning.

## Results summary
Results indicate that the factors included in the regression models were not adequate predictors for hotel guest ratings. Multinomial and ordinal logistic regression models were the best performers at 0.40 mean accuracy. The models could be improved, most likely, by including more numeric variables that provide quantitative description of the hotels in the study (the databases with these information are not free). Also, model performance could be improved if the ratings for the hotels were not skewed to the left (the dataset used shows that most of the guest ratings are 4's and 5's).

On the other hand, the model developed to predict hotel guest rating from guest comments performed at 75% prediction accuracy. Words most *frequently* associated with hotel guest ratings were adjectives (great, good, dirty, clean) and meals (breakfast). However, by using TF-IDF, the *important* words separating the different ratings were different.

## Presentation
A summary of the findings are presented in the Google Slide Deck found [here](https://docs.google.com/presentation/d/1V_myvfP2MIwZeA6Qpv7WwfCHxwd0o0B3AISVZQSu5tM/edit#slide=id.g522128fe48_1_61). A pdf version of the slides is found [here](https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/ML_Hotel_Ratings.pdf).