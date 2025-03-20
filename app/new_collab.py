from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine
from configuration import Config


# Подключение к базе данных
engine = create_engine(Config.DATABASE_URI, echo=False)

# Загружаем данные
ratings_df = pd.read_sql("SELECT user_id, recipe_id, rating FROM user_ratings", engine)
# Загрузка данных
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'recipe_id', 'rating']], reader)

# Разделение данных на обучающую и тестовую выборки
trainset, testset = train_test_split(data, test_size=0.2)

# Обучение модели SVD (Singular Value Decomposition)
model = SVD()
model.fit(trainset)

# Предсказание рейтинга для пользователя и рецепта
user_id = 1
recipe_id = 100
predicted_rating = model.predict(user_id, recipe_id).est
print(f"Предсказанный рейтинг: {predicted_rating}")
from surprise import accuracy

# Оценка коллаборативной фильтрации
predictions = model.test(testset)
print(f"RMSE: {accuracy.rmse(predictions)}")