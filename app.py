from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# load dataset
movies = pd.read_csv("./dataset/movies.csv")

# tfidf dan cosin similarity
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosin_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
movies_indicies = pd.Series(movies.index, index=movies["title"]).drop_duplicates()


# fungsi rekomendasi
def recommend_movies(title):
    if title not in movies_indicies:
        return []

    idx = movies_indicies[title]
    sim_scores = list(enumerate(cosin_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sim_scores[1:6]]
    return movies["title"].iloc[recommended_indices].tolist()


# endpoint rekomendasi
@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Parameter 'title' diperlukan!"}), 400
    recommendations = recommend_movies(title)
    if not recommendations:
        return jsonify({"error": "Film tidak ditemukan"}), 404
    return jsonify({"title": title, "recommendations": recommendations})


if __name__ == "__main__":
    app.run(debug=True)
