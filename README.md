# Exploratory Analysis of Hotel Ratings Data using Machine Learning Algorithms
## Introduction
What compels people to take the time to write a review about the hotels they stayed at? What factors influence their ratings? These are the kinds of information property owners and managers can glean insights on to improve services and facilities, to determine the sentiments of their consumer niche, and to perform at par or better than their competition.

In this project, data on consumer reviews and other data of over 1,800 hotels will be used to model the factors that lead to hotel ratings and to find out what makes hotel visitors tick through machine learning approaches. It is possible that through this work, new insights may be generated for benchmarking hotel performance.

### Objectives
1. To create a model that can be used to predict hotel ratings given by visitors based on various variables.
2. To obtain sentiment insights from the visitors who provided information about their experiences in the various properties.

## Jupyter notebooks
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
        <td>Natural language processing (NLP) on each hotel review
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
</table>

## Database
The data extracted and transformed in Jupyter notebooks were saved in <a href = "https://github.com/janelcv/Hotel_Rating_Analysis/blob/master/Data/Hotels.db">Hotels.db</a>. It contains three tables:
1. <b>metadata.</b> Raw data.
2. <b>metadata2.</b> Dummy variables and nearest-airport distance indicated.
3. <b>ratings.</b> Scores and reviews obtained from hotel visitors.
4. <b>alldata.</b> Merged data from metadata2 and ratings; for machine learning.