from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router as recipe_router

app = FastAPI()

# Подключаем маршруты
app.include_router(recipe_router)

# Настроим CORS
origins = [
    "http://localhost:5173",  # Разрешаем доступ с фронт-энд сервера
    "http://localhost",       # Можно добавить другие источники, если нужно
    "http://127.0.0.1",       # Допустим, если ваш фронт работает на 127.0.0.1
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Разрешенные источники
    allow_credentials=True,
    allow_methods=["*"],    # Разрешаем все HTTP методы
    allow_headers=["*"],    # Разрешаем все заголовки
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=['*']
# )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Recipe Recommendation System!"}

