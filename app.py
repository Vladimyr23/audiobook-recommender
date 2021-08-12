from flask import Flask, render_template, request

#import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# webapp
app = Flask(__name__, template_folder='./')


# model = model

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
# @cross_origin()
def recommend():
    features = [str(x) for x in request.form.values()]
    final_features = []

    for x in features:
        if x != 'skip':
            soup = " ".join(["".join(n.split()) for n in x.lower().split(',')])
            final_features.append(soup)

    audible_data = pd.read_csv("Audible_Dataset_final_TGH.csv",
                               encoding='latin1')

    #audible_data = audible_data.iloc[0:10000, :]

    # Remove all 'Categories', and 'Book Narrator' NaN records
    audible_data = audible_data[audible_data['Categories'].notna()]
    audible_data = audible_data[audible_data['Book Narrator'].notna()]

    # Selecting 4 columns: Title, Author, Narrator,Categories(Genre)
    audible_data = audible_data[['Book Title', 'Book Author', 'Book Narrator', 'Categories']]

    # lower case and split on commas or &-sign 'Categories'
    audible_data['Categories'] = audible_data['Categories'].map(
        lambda x: x.lower().replace(' &', ',').replace('genre', '').split(','))
    # Book Author
    audible_data['Book Author'] = audible_data['Book Author'].map(lambda x: x.lower().replace(' ', '').split(' '))
    # Book Narrator
    audible_data['Book Narrator'] = audible_data['Book Narrator'].map(lambda x: x.lower().replace(' ', '').split(' '))

    for index, row in audible_data.iterrows():
        # row['Book Narrator'] = [x.replace(' ','') for x in row['Book Narrator']]
        row['Book Author'] = ''.join(row['Book Author'])

    # make 'Book Title' as an index
    audible_data.set_index('Book Title', inplace=True)

    audible_data['bag_of_words'] = ''
    for index, row in audible_data.iterrows():
        words = ''
        for col in audible_data.columns:
            if col != 'Book Author':
                words = words + ' '.join(row[col]) + ' '
            else:
                words = words + row[col] + ' '
        row['bag_of_words'] = words

    audible_data.drop(columns=[x for x in audible_data.columns if x != 'bag_of_words'], inplace=True)

    recommendation_movies = []

    # adding the new row to the dataset
    audible_data.loc['USER'] = [" ".join(final_features)]

    # Vectorizing the entire matrix as described above!
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(audible_data['bag_of_words'])

    # running pairwise cosine similarity
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)  # getting a similarity matrix

    # gettin the index of the movie that matches the title
    indices = pd.Series(audible_data.index)
    idx = indices[indices == 'USER'].index[0]
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim2[idx]).sort_values(ascending=False)
    # getting the indexes of the 10 most similar movies
    top_5_indexes = list(score_series.iloc[1:6].index)
    print(top_5_indexes)
    # populating the list with the titles of the best 10 matching movies
    for i in top_5_indexes:
        recommendation_movies.append(list(audible_data.index)[i])
    recommend_str = ', '.join(recommendation_movies)

    return render_template('index.html',
                           recommendation_text=('Here are some audiobooks for you: ', recommend_str),
                           genre=request.form["book genre"],
                           author=request.form["book author"],
                           narrator=request.form["book narrator"])


if __name__ == '__main__':
    app.run(debug=True)
