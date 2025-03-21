import pickle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine
from configuration import Config
from sklearn.model_selection import train_test_split

# Подключение к базе данных
engine = create_engine(Config.DATABASE_URI, echo=False)

# Загружаем данные
ratings_df = pd.read_sql("SELECT user_id, recipe_id, rating FROM user_ratings", engine)

# Создаем разреженную матрицу
user_ids = ratings_df["user_id"].astype("category").cat.codes
recipe_ids = ratings_df["recipe_id"].astype("category").cat.codes
ratings = ratings_df["rating"].astype(float)

sparse_matrix = coo_matrix((ratings, (user_ids, recipe_ids)))

# Сохранение данных для использования в приложении
with open("ratings_data.pkl", "wb") as f:
    pickle.dump({
        "ratings_df": ratings_df,
        "user_ids": ratings_df["user_id"].astype("category").cat.categories,
        "recipe_ids": ratings_df["recipe_id"].astype("category").cat.categories,
        "sparse_matrix": sparse_matrix  # Используем существующую sparse_matrix
    }, f)

# Обучаем ALS
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=50)
model.fit(sparse_matrix)
# Сохранение модели
model.save("als_model.npz")

def recommend_recipes(user_id, n=5):


    if user_id not in ratings_df["user_id"].values:
        popular_recipes = ratings_df["recipe_id"].value_counts().index.tolist()
        return popular_recipes[:n]
    
    user_code = ratings_df["user_id"].astype("category").cat.categories.get_loc(user_id)
    sparse_csr = sparse_matrix.tocsr()
    user_interactions = sparse_csr[user_code]
    
    recommended = model.recommend(user_code, user_interactions, N=n)
    recommended_recipe_ids = [ratings_df["recipe_id"].iloc[int(rec[0])] for rec in recommended]
    
    return recommended_recipe_ids
# def recommend_recipes(user_id, n=5):
    if user_id not in train_df["user_id"].values:
        # Возвращаем популярные рецепты для новых пользователей
        popular_recipes = train_df["recipe_id"].value_counts().index.tolist()
        return popular_recipes[:n]
    
    user_code = train_df["user_id"].astype("category").cat.categories.get_loc(user_id)
    sparse_csr = sparse_matrix.tocsr()
    user_interactions = sparse_csr[user_code]
    
    recommended = model.recommend(user_code, user_interactions, N=n)
    recommended_recipe_ids = [train_df["recipe_id"].iloc[int(rec[0])] for rec in recommended]
    
    return recommended_recipe_ids

# Оценка модели

def precision_at_k(actual, predicted, k):
    return sum(1 for rec in predicted[:k] if rec in actual) / k

def recall_at_k(actual, predicted, k):
    return sum(1 for rec in predicted[:k] if rec in actual) / len(actual) if actual else 0

def f1_at_k(actual, predicted, k):
    p = precision_at_k(actual, predicted, k)
    r = recall_at_k(actual, predicted, k)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def map_at_k(actual, predicted, k):
    k = min(k, len(predicted))
    if k == 0:
        return 0
    return sum(precision_at_k(actual, predicted, i + 1) for i in range(k) if predicted[i] in actual) / k

def ndcg_at_k(actual, predicted, k):
    def dcg(recs, relevant, k):
        k = min(k, len(recs))
        if k == 0:
            return 0
        return sum(int(recs[i] in relevant) / np.log2(i + 2) for i in range(k))

    idcg = dcg(sorted(actual, reverse=True), actual, k)
    return dcg(predicted, actual, k) / idcg if idcg > 0 else 0

# Пример оценки для одного пользователя
user_test = 12345  # Пример пользователя
actual_recipies = ratings_df[ratings_df["user_id"] == user_test]["recipe_id"].tolist()
predicted_recipies = recommend_recipes(user_test, 5)
k = min(5, len(predicted_recipies))

print(f"Актуальные рецепты: {actual_recipies}")
print(f"Рекомендованные рецепты: {predicted_recipies}")
print(f"Precision@5: {precision_at_k(actual_recipies, predicted_recipies, k):.4f}")
print(f"Recall@5: {recall_at_k(actual_recipies, predicted_recipies, k):.4f}")
print(f"F1@5: {f1_at_k(actual_recipies, predicted_recipies, k):.4f}")
print(f"MAP@5: {map_at_k(actual_recipies, predicted_recipies, k):.4f}")
print(f"NDCG@5: {ndcg_at_k(actual_recipies, predicted_recipies, k):.4f}")