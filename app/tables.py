from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float, Boolean, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import Table
from sqlalchemy import DateTime, Date
from math import ceil
from configuration import Config


engine = create_engine(Config.DATABASE_URI, echo=False)
Base = declarative_base()


class Recipes(Base):
    __tablename__ = 'recipes_povarenok'
    id = Column(Integer, autoincrement=True, primary_key=True)
    url = Column(String)
    name = Column(String)
    ingredients = Column(String)
    ingredients_list = Column(String)


class RecipesData(Base):
    __tablename__ = 'recipes_data_copy'
    id = Column(Integer, autoincrement=True, primary_key=True)
    #Title,Category,Cooking Time,Likes,Ingredients,Calories,Protein,Fat,Carbs,URL
    title = Column(String)
    category = Column(String)
    cooking_time = Column(Integer)
    likes = Column(Integer)
    dislikes = Column(Integer)
    ingredients = Column(String)
    calories = Column(Float)
    protein = Column(Float)
    fat = Column(Float)
    carbs = Column(Float)
    url = Column(String)
    ingredients_list = Column(String)


class Ingredients(Base):
    __tablename__ = 'ingredients_povarenok'
    id = Column(Integer, autoincrement=True, primary_key=True)
    name = Column(String)
    # formated_name = Column(String)


class RecipesIngredients(Base):
    __tablename__ = 'recipes_ingredients'
    id = Column(Integer, autoincrement=True, primary_key=True)
    recipe_id = Column(Integer)
    ingredient_id = Column(Integer)


class IngredientsVector(Base):
    __tablename__ = 'ingredients_vector'
    id = Column(Integer, autoincrement=True, primary_key=True)
    recipe_id = Column(Integer)
    vector = Column(Integer)