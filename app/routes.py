from fastapi import APIRouter, HTTPException, Query
from typing import List
import models as m
import services as serv
from typing import List
import requests
from bs4 import BeautifulSoup

router = APIRouter()

@router.get("/api/ingredients", response_model=List[m.Ingredient])
def get_ingredients():
    """Эндпоинт для получения всех возможных ингредиентов."""
    ingredients = serv.get_all_ingredients()

    if not ingredients:
        raise HTTPException(status_code=404, detail="Список ингредиентов пуст")
    
    return ingredients


@router.post("/api/recipes")
def get_recipes(request: m.IngredientsRequest):  
    print(f"Получен запрос: {request}")

    if not request.ingredients:
        raise HTTPException(status_code=400, detail="Список ингредиентов не может быть пустым")

    matching_recipes = serv.get_all_recipes(request.ingredients)
    return matching_recipes


@router.post("/api/parse_recipe")
def parse_recipe(request: m.RecipeURLRequest):
    """Эндпоинт для парсинга рецепта по ссылке"""

    try:
        response = requests.get(request.url, timeout=5)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Не удалось загрузить страницу")

        data = parse_recipe(request)
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при парсинге: {str(e)}")
    


@router.get("/api/recommendations", response_model=List[m.Recipe])
def get_recommendations(user_id: int = Query(..., description="ID пользователя")):
    """
    Получить персонализированные рекомендации для пользователя.
    """
    print(f"Получен запрос для user_id: {user_id}")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id не может быть пустым")

    try:
        # Логируем входные данные
        print(f"Запрос для user_id: {user_id}")
        print(f'Тип данных: {type(user_id)}')
        
        # Получаем рекомендации
        recommended_recipes = serv.get_personal_recommendation(user_id)
        return recommended_recipes
    except Exception as e:
        # Логируем ошибку
        print(f"Ошибка при получении рекомендаций: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении рекомендаций: {str(e)}")


# @router.post("/api/recommendations")
# def get_recommendations(request: m.UserRequest):
#     print(f"Получен запрос: {request}")

#     if not request.ingredients:
#         raise HTTPException(status_code=400, detail="Список ингредиентов не может быть пустым")

#     matching_recipes = serv.get_personal_recommendation(request.id)
#     return matching_recipes