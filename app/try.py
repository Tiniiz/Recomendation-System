# import asyncio
# import aiohttp
# import pandas as pd
# import multiprocessing
# import json
# from tqdm import tqdm
# from sqlalchemy import create_engine
# from configuration import Config
# from bs4 import BeautifulSoup

# engine = create_engine(Config.DATABASE_URI, echo=False)

def get_recipes():

    query = """SELECT * FROM recipes_povarenok ORDER BY id"""
    df = pd.read_sql(query, engine)
    return df


# # Количество одновременных запросов
# MAX_CONCURRENT_REQUESTS = 50  

# # Заголовки для маскировки под браузер
# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# }

# async def fetch(session, url, pbar):
#     """Загружает страницу с обработкой ошибок и 429"""
#     for attempt in range(5):  # 5 попыток
#         try:
#             async with session.get(url, headers=HEADERS, timeout=10) as response:
#                 if response.status == 429:  # Слишком много запросов
#                     wait_time = 5 * (attempt + 1)
#                     print(f"⚠ Получен 429, жду {wait_time} сек...")
#                     await asyncio.sleep(wait_time)
#                     continue
#                 response.raise_for_status()
#                 text = await response.text()
#                 pbar.update(1)  # Обновляем прогресс
#                 return text
#         except asyncio.TimeoutError:
#             print(f"⏳ Тайм-аут {url}, попытка {attempt + 1}")
#         except aiohttp.ClientError as e:
#             print(f"⚠ Ошибка {url}: {e}")
#         await asyncio.sleep(1)  # Короткая пауза перед повтором
#     return None  # Если все попытки провалены

# async def parse_page(url, session, pbar):
#     """Парсит страницу"""
#     html = await fetch(session, url, pbar)
#     if html is None:
#         return None

#     soup = BeautifulSoup(html, "html.parser")
#     title = soup.find('h1', {'itemprop': "name"})
#     title = title.text.strip() if title else 'Нет названия'
    
#     ingredients = [item.text.strip() for item in soup.select('.ingredients-bl li')] or ['Не указано']
#     cook_time = soup.find('time', {'itemprop': 'totalTime'})
#     cook_time = cook_time.text.strip() if cook_time else 'Не указано'
    
#     steps = []
#     for step in soup.select('.cooking-bl'):
#         img_tag = step.find('img')
#         img_url = img_tag['src'] if img_tag else None
#         text = step.find('div p')
#         text = text.text.strip() if text else None
#         steps.append({"image_url": img_url, "text": text})

#     return {
#         "name": title,
#         "details": "\n".join(ingredients),
#         "cook_time": cook_time,
#         "nutrition_data": json.dumps({}, ensure_ascii=False),
#         "steps": json.dumps(steps, ensure_ascii=False)
#     }

# async def process_pages(urls):
#     """Запускает парсинг с большим количеством запросов"""
#     connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT_REQUESTS)  # Ограничение запросов к одному хосту
#     async with aiohttp.ClientSession(connector=connector) as session:
#         with tqdm(total=len(urls), desc="Прогресс", ncols=80) as pbar:
#             tasks = [parse_page(url, session, pbar) for url in urls]
#             return await asyncio.gather(*tasks)

# if __name__ == "__main__":
#     df = get_recipes()  # Файл со списком ссылок
#     urls = df['url'].tolist()

#     parsed_data = asyncio.run(process_pages(urls))

#     new_df = pd.DataFrame([item for item in parsed_data if item])
#     new_df.to_sql('recipes_details_povarenok', engine, if_exists='replace', index=False)

#     print("Готово! Данные сохранены.")


import numpy as np
import pandas as pd
from configuration import Config
from sqlalchemy import create_engine

# Подключение к базе данных
engine = create_engine(Config.DATABASE_URI, echo=False)
# Параметры данных
num_users = 50000
num_recipes = 146504
min_ratings_per_user = 100  # Минимум 100 оценок на пользователя
min_ratings_per_recipe = 50  # Минимум 50 оценок на рецепт
total_ratings = 10000000  # Общее количество оценок

# Генерация случайных данных
np.random.seed(42)  # Для воспроизводимости

# Создаем списки user_id и recipe_id
user_ids = np.random.choice(np.arange(1, num_users + 1), size=total_ratings, replace=True)
recipe_ids = np.random.choice(np.arange(1, num_recipes + 1), size=total_ratings, replace=True)

# Генерация оценок (от 1 до 5)
ratings = np.random.randint(1, 6, size=total_ratings)

# Создаем DataFrame
ratings_df = pd.DataFrame({
    "user_id": user_ids,
    "recipe_id": recipe_ids,
    "rating": ratings
})

# Удаляем дубликаты (если один пользователь оценил один рецепт несколько раз)
ratings_df = ratings_df.drop_duplicates(subset=["user_id", "recipe_id"])

# Проверяем, что у каждого пользователя и рецепта достаточно оценок
user_counts = ratings_df.groupby("user_id").size()
recipe_counts = ratings_df.groupby("recipe_id").size()

# Фильтруем пользователей и рецепты с недостаточным количеством оценок
valid_users = user_counts[user_counts >= min_ratings_per_user].index
valid_recipes = recipe_counts[recipe_counts >= min_ratings_per_recipe].index
ratings_df = ratings_df[ratings_df["user_id"].isin(valid_users) & ratings_df["recipe_id"].isin(valid_recipes)]

# Проверяем итоговые данные
print(f"Количество пользователей: {ratings_df['user_id'].nunique()}")
print(f"Количество рецептов: {ratings_df['recipe_id'].nunique()}")
print(f"Количество оценок: {len(ratings_df)}")
print(f"Среднее количество оценок на пользователя: {ratings_df.groupby('user_id').size().mean()}")
print(f"Среднее количество оценок на рецепт: {ratings_df.groupby('recipe_id').size().mean()}")


ratings_df.to_sql('user_ratings', engine, if_exists='replace', index=False)