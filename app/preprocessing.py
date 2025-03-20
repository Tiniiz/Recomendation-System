import json
import pickle
from bs4 import BeautifulSoup
from fastapi import HTTPException
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from configuration import Config
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker
import tables as t
from gensim.models import Word2Vec
import numpy as np
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import preprocessing as pcg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from GCN import GCN
import models as m
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares


engine = create_engine(Config.DATABASE_URI, echo=False)


# def recommend_recipe(user_ingredients, knn_model, recipes_data):
#     # Преобразуем данные пользователя в формат для модели
#     user_input_vector = preprocess_user_input(user_ingredients)
    
#     # Используем модель KNN для поиска ближайших рецептов
#     distances, indices = knn_model.kneighbors([user_input_vector])
    
#     # Получаем результаты
#     recommendations = []
#     for idx, dist in zip(indices[0], distances[0]):
#         recipe = recipes_data.iloc[idx]  # Используем .iloc для доступа по индексу
#         # recommendations.append({
#         #     'recipe': recipe,
#         #     'recipe_name': idx,
#         #     'distance': dist
#         # })
#         recommendations.append(int(idx))
    
#     return recommendations


# def preprocess_user_input(ingredients):
#     # Загружаем модель Word2Vec
#     vectorizer = Word2Vec.load("word2vec_model.model")
#     ingredients = [word.lower() for word in ingredients]
    
#     # Векторизуем ингредиенты
#     vectors = [vectorizer.wv[word] for word in ingredients if word in vectorizer.wv]
    
#     # Если векторы найдены, усредняем их
#     if vectors:
#         ingredient_vectors = np.mean(vectors, axis=0)
#     else:
#         # Если ни один ингредиент не найден, возвращаем нулевой вектор
#         ingredient_vectors = np.zeros(vectorizer.vector_size)

#     print(len(ingredient_vectors))
    
#     return ingredient_vectors


def get_recipes():

    query = """SELECT * FROM recipes_povarenok ORDER BY id"""
    df = pd.read_sql(query, engine)
    return df


def get_ingredients():

    query = """SELECT * FROM ingredients_povarenok ORDER BY id"""
    df = pd.read_sql(query, engine)
    return df


def get_vectors():

    query = """SELECT * FROM ingredients_vector ORDER BY id"""
    df = pd.read_sql(query, engine)
    df = df.drop('id', axis=1)
    return df


def predict_recipes(ingredient_list, top_n=5):
    embedding_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем данные
    recipes = pcg.get_recipes()
    recipes['ingredients_list'] = recipes['ingredients_list'].str.split(', ')


    # Собираем уникальные ингредиенты
    ingredients = list(set(ingr for ings in recipes['ingredients_list'] for ingr in ings))

    # Создаем словари для маппинга имен в индексы
    ingredient_to_idx = {ing: idx for idx, ing in enumerate(ingredients)}
    # Обновляем словарь для маппинга рецептов на индексы, используя 'recipe_id'
    recipe_to_idx = {rec_id: idx + len(ingredients) for idx, rec_id in enumerate(recipes['id'])}


    # Список всех узлов (ингредиенты + рецепты)
    node_labels = ingredients + recipes['id'].tolist()

    # Создаем ребра (связи между ингредиентами и рецептами)
    edges = []
    for _, row in recipes.iterrows():
        recipe_idx = recipe_to_idx[row['id']]  # Используем 'recipe_id'
        for ing in row['ingredients_list']:
            ing_idx = ingredient_to_idx[ing]
            edges.append([ing_idx, recipe_idx])  # Ингредиент -> Рецепт
            edges.append([recipe_idx, ing_idx]) 

    # Преобразуем ребра в тензор
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # edge_index = augment_graph_with_edge_dropout(edge_index, dropout_rate=0.2)

    # Создаем граф
    data = Data(edge_index=edge_index, num_nodes=len(node_labels))
    data.x = torch.randn(len(node_labels), embedding_dim) 
    # Загрузка модели
    # model = GCN(in_channels=embedding_dim, hidden_channels=8, num_layers=2).to(device)
    model = GCN(num_node_features=embedding_dim, hidden_channels=8, num_classes=2).to(device)
    model.load_state_dict(torch.load('gcn_model1.pth'))
    model.eval()
    
    # Преобразуем ингредиенты в индексы
    ingredient_indices = [ingredient_to_idx[ing] for ing in ingredient_list if ing in ingredient_to_idx]
    
    if not ingredient_indices:
        print("Нет известных ингредиентов в списке!")
        return []
    
    # Найдём связанные рецепты
    recipe_candidates = set()
    for edge in edge_index.t().tolist():
        if edge[0] in ingredient_indices:
            recipe_candidates.add(edge[1])  # Добавляем рецепты
    
    recipe_candidates = list(recipe_candidates)
    if not recipe_candidates:
        print("Нет рецептов с такими ингредиентами!")
        return []

    # Создаем входные данные для модели
    recipe_indices = torch.tensor(recipe_candidates, dtype=torch.long).to(device)
    print(data)  # Выведет всю структуру data
    print(data.x)  # Проверит, есть ли x
    # Переносим данные на нужное устройство
    data = data.to(device)  # Переносим весь объект Data
    recipe_indices = recipe_indices.to(device)  # Переносим индексы

    x_input = data.x[recipe_indices]  # Берём эмбеддинги только для этих узлов
    edge_subset = torch.stack([recipe_indices, recipe_indices])  # Заглушка, если без графа
    
    with torch.no_grad():
        out = model(x_input, edge_subset)
    
    # Выбираем рецепты с наибольшими вероятностями
    probs = torch.exp(out[:, 1])  # Берем вероятность класса "рецепт"
    top_indices = torch.argsort(probs, descending=True)[:top_n]
    
    return [node_labels[recipe_candidates[i]] for i in top_indices]




def recommend_recipes(user_id, n=5):

    try:
        # Загрузка данных при запуске приложения
        with open("ratings_data.pkl", "rb") as f:
            data = pickle.load(f)
            ratings_df = data["ratings_df"]
            user_ids = data["user_ids"]
            recipe_ids = data["recipe_ids"]
            sparse_matrix = data["sparse_matrix"].tocsr()  # Преобразуем в CSR для эффективности
    except FileNotFoundError:
        raise ValueError(f"Файл с данными для пользователя {user_id} не найден")
    except json.JSONDecodeError:
        raise ValueError(f"Файл с данными для пользователя {user_id} поврежден")

    # Загрузка модели
    model = AlternatingLeastSquares()
    model = model.load("als_model.npz")

    if user_id not in ratings_df["user_id"].values:
        # Если пользователя нет, возвращаем популярные рецепты
        popular_recipes = ratings_df["recipe_id"].value_counts().index.tolist()
        return popular_recipes[:n]
    
    # Получаем код пользователя
    user_code = ratings_df["user_id"].astype("category").cat.categories.get_loc(user_id)
    
    # Получаем рекомендации
    recommended = model.recommend(user_code, sparse_matrix[user_code], N=n)
    recommended_recipe_ids = [ratings_df["recipe_id"].iloc[int(rec[0])] for rec in recommended]
    
    return recommended_recipe_ids