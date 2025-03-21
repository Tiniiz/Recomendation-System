from typing import List
import json
from sqlalchemy import create_engine
import models as m
import tables as t
from configuration import Config
from sqlalchemy.orm import sessionmaker
import preprocessing as pcg
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import average_precision_score, pairwise_distances
from sklearn.model_selection import train_test_split
import joblib
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sqlalchemy import create_engine
from configuration import Config
from sklearn.model_selection import train_test_split


engine = create_engine(Config.DATABASE_URI, echo=False)


def get_all_ingredients() -> List[m.Ingredient]:
    """Функция для поиска ингредиентов"""

    Session = sessionmaker(bind=engine)
    session = Session()

    ingredients = session.query(t.Ingredients).all()

    session.close()

    return [m.Ingredient(name=ingr.name) for ingr in ingredients]


def get_all_recipes(ingredients: List[m.IngredientsRequest]) -> List[m.Recipe]:

    Session = sessionmaker(bind=engine)
    session = Session()

    # Рекомендация рецептов
    predicted_recipes = pcg.predict_recipes(ingredients)

    Session = sessionmaker(bind=engine)
    session = Session()

    recipes = session.query(t.RecipesData).filter(t.RecipesData.id.in_(predicted_recipes)).all()

    return [m.Recipe(name=recipe.title,
        category=recipe.category,
        cooking_time=recipe.cooking_time,
        ingredients=recipe.ingredients, 
        calories=recipe.calories,
        protein=recipe.protein,
        fat=recipe.fat,
        carbs=recipe.carbs,
        url=recipe.url,
        ingredients_list=recipe.ingredients_list) for recipe in recipes]

def clean_text(text):
    return text.strip() if text and text.strip() else None


def parse_page(request: str):
    try:
        response = requests.get(request, timeout=5)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Не удалось загрузить страницу")

        soup = BeautifulSoup(response.text, "html.parser")

        # Извлечение названия рецепта
        title = soup.find('h1', {'itemprop': "name"})
        title = title.text.strip() if title else 'Нет названия'
        
        # Извлечение ингредиентов
        ingredients = []
        ingredients_section = soup.find(class_='ingredients-bl')
        if ingredients_section:
            for item in ingredients_section.find_all('li'):
                ingredients.append(item.text.strip())
    
        # Извлечение времени приготовления
        cook_time_tag = soup.find('time', {'itemprop': 'totalTime'})
        cook_time = cook_time_tag.text.strip() if cook_time_tag else 'Не указано'
        
        # Извлечение данных о калорийности
        nutrition_data = {
            "Готового блюда": {},
            "Порции": {},
            "100 г блюда": {}
        }

        rows = soup.find_all('tr')
        current_section = None

        for row in rows:
            if row.find('td', class_='nae-title'):
                current_section = row.find('strong').text.strip()
            else:
                cells = row.find_all('td')
                if len(cells) == 4 and current_section:
                    nutrition_data[current_section] = {
                        "ккал": cells[0].find('strong').text.strip() if cells[0].find('strong') else 'Не указано',
                        "белки": cells[1].find('strong').text.strip() if cells[1].find('strong') else 'Не указано',
                        "жиры": cells[2].find('strong').text.strip() if cells[2].find('strong') else 'Не указано',
                        "углеводы": cells[3].find('strong').text.strip() if cells[3].find('strong') else 'Не указано'
                    }
        nutrition_data_str = json.dumps(nutrition_data, ensure_ascii=False)

        # Извлечение шагов приготовления
        steps = soup.find_all('li', class_='cooking-bl')
        data = []

        for step in steps:
            img_tag = step.find('img')
            img_url = img_tag['src'] if img_tag else None
            if img_url and not img_url.startswith("http"):
                img_url = request.url.rsplit("/", 1)[0] + "/" + img_url  # Делаем полный путь


            text = step.find('div').find('p').text.strip() if step.find('div') and step.find('div').find('p') else None

            data.append({
                "image_url": img_url,
                "text": text
            })

        # Формирование результата
        steps_text = "\n".join([step['text'] for step in data if step['text']])
        details = f"Ингредиенты:\n" + "\n".join(ingredients) + "\n\nШаги приготовления:\n" + steps_text

        steps_str = json.dumps(data, ensure_ascii=False)


        return m.RecipeFullDetails(
            name=title,
            details=details,
            cook_time=cook_time,
            nutrition_data=nutrition_data_str,
            steps=steps_str 
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при парсинге: {str(e)}")

    #     const title = $('h1[itemprop="name"]').text().trim();
    # const cookingTime = $('time[itemprop="totalTime"]').attr('datetime')?.match(/PT(\d+)M/)?.[1] 
    # ?? parseInt($('time[itemprop="totalTime"]').attr('datetime')?.match(/PT(\d+)H/)?.[1] || '0') * 60 
    # ?? '0';
    # const likes = $('span.i-views').text().trim();

    # const ingredients :string[] = [];
    # $('li[itemprop="recipeIngredient"]').each(function() {
    #     const name = $(this).find('span').first().text().trim();
    #     const amount = $(this).find('span').last().text().trim();
    #     ingredients.push(`${name} — ${amount}`);
    # });
    # const ingredientsString = ingredients.join(', ');

    # const nutritionRows = $("tr:contains('100 г блюда')").next();
    # const calories = nutritionRows.find("td:contains('ккал') strong").text().trim();
    # const protein = nutritionRows.find("td:contains('белки') strong").text().trim();
    # const fat = nutritionRows.find("td:contains('жиры') strong").text().trim();
    # const carbs = nutritionRows.find("td:contains('углеводы') strong").text().trim();
    
    # const category = $('span[itemprop="recipeCategory"]').first().text().trim();


def get_personal_recommendation(user: int) -> List[m.Recipe]:

    Session = sessionmaker(bind=engine)
    session = Session()

    # Рекомендация рецептов
    predicted_recipes = pcg.recommend_recipes(user)

    print("Predicted Recipes:", predicted_recipes)
    print("Types:", [type(recipe) for recipe in predicted_recipes])

    # Step 3: Query the database
    Session = sessionmaker(bind=engine)
    session = Session()

    recipes = session.query(t.RecipesData).filter(t.RecipesData.id.in_(predicted_recipes)).all()

    # Step 4: Process the results
    for recipe in recipes:
        print(recipe.title, recipe.category)

    return [m.Recipe(name=recipe.title,
        category=recipe.category,
        cooking_time=recipe.cooking_time,
        ingredients=recipe.ingredients, 
        calories=recipe.calories,
        protein=recipe.protein,
        fat=recipe.fat,
        carbs=recipe.carbs,
        url=recipe.url,
        ingredients_list=recipe.ingredients_list) for recipe in recipes]


