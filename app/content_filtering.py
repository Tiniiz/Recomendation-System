# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# from sqlalchemy import create_engine
# from configuration import Config
# from sklearn.model_selection import train_test_split
# from scipy.sparse import csr_matrix

# # Подключение к базе данных
# engine = create_engine(Config.DATABASE_URI, echo=False)

# def get_recipes():

#     query = """SELECT * FROM recipes_povarenok ORDER BY id"""
#     df = pd.read_sql(query, engine)
#     return df

# # Загружаем данные
# ratings_df = pd.read_sql("SELECT user_id, recipe_id, rating FROM user_ratings", engine)

# df = get_recipes()
# tfidf = TfidfVectorizer()
# # movies['overview'] = movies['overview'].fillna('')
# overview_matrix = tfidf.fit_transform(df['ingredients_list'])
# overview_matrix = csr_matrix(overview_matrix)
# similarity_matrix = linear_kernel(overview_matrix,overview_matrix)
# mapping = pd.Series(df.index,index = df['title'])
# print(mapping)

# def recommend_movies(recipe_input):
#     recipe_index = mapping[recipe_input]
#     similarity_score = list(enumerate(similarity_matrix[recipe_index]))
#     similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
#     similarity_score = similarity_score[1:15]
#     recipe_indices = [i[0] for i in similarity_score]
#     return (df['title'].iloc[recipe_indices])

# print(recommend_movies('Рыба «Красное и белое»'))


# Загрузка данных
df = get_recipes()
ratings_df = pd.read_sql("SELECT user_id, recipe_id, rating FROM user_ratings", engine)

# Коллаборативная фильтрация
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'recipe_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# Контентный подход
tfidf = TfidfVectorizer(stop_words='english')
overview_matrix = tfidf.fit_transform(df['ingredients_list'])
similarity_matrix = linear_kernel(overview_matrix, overview_matrix)

def recommend_recipes_content_based(recipe_title, top_n=10):
    recipe_index = df[df['title'] == recipe_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[recipe_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n+1]  # Исключаем сам рецепт
    recipe_indices = [i[0] for i in similarity_scores]
    return df['title'].iloc[recipe_indices]

# Гибридный подход
def hybrid_recommendation(user_id, recipe_title, alpha=0.5):
    recipe_id = df[df['title'] == recipe_title]['id'].values[0]
    collaborative_score = model.predict(user_id, recipe_id).est
    content_based_recommendations = recommend_recipes_content_based(recipe_title)
    content_based_score = content_based_recommendations.index.get_loc(recipe_id) if recipe_id in content_based_recommendations.index else 0
    hybrid_score = alpha * collaborative_score + (1 - alpha) * content_based_score
    return hybrid_score

# Пример использования
user_id = 1
recipe_title = 'Рыба «Красное и белое»'
print(f"Гибридный рейтинг: {hybrid_recommendation(user_id, recipe_title)}")