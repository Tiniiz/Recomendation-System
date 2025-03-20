from pydantic import BaseModel
from typing import List

# Модель рецепта
class Recipe(BaseModel):
    name: str
    # category: str
    # cooking_time: int
    # ingredients: str 
    # calories: float
    # protein: float
    # fat: float
    # carbs: float
    url: str
    ingredients_list: str
    ingredients: str 


# Модель ингредиента
class Ingredient(BaseModel):
    name: str


# Модель для запроса
class IngredientsRequest(BaseModel):
    ingredients: List[str]  # Список строк


class RecipeURLRequest(BaseModel):
    url: str


class RecipeFullDetails(BaseModel):
    name: str
    details: str
    cook_time: str
    nutrition_data: str
    steps: str


class UserRequest(BaseModel):
    id: int