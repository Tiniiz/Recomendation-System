import { useState, useEffect } from "react";
import {
  TextField,
  Button,
  Card,
  CardContent,
  Typography,
  Container,
  Box,
  Grid,
  Autocomplete,
  Chip,
  CircularProgress,
} from "@mui/material";
import { useNavigate } from "react-router-dom"; // Для навигации

interface Recipe {
  id: number;
  name: string;
  ingredients: string;
  url: string; // Добавляем поле URL
}

interface Ingredient {
  name: string;
}

export default function RecipeFinder() {
  const [ingredients, setIngredients] = useState<{ name: string }[]>([]);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [availableIngredients, setAvailableIngredients] = useState<Ingredient[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userId, setUserId] = useState<string>(""); // Состояние для user_id
  const [recommendedRecipes, setRecommendedRecipes] = useState<Recipe[]>([]); // Состояние для рекомендаций
  const navigate = useNavigate(); // Хук для навигации

  // Загрузка ингредиентов с сервера
  useEffect(() => {
    const fetchIngredients = async () => {
      setError(null); // Сброс ошибки перед загрузкой
      try {
        const response = await fetch("http://localhost:8000/api/ingredients");
        if (!response.ok) {
          throw new Error("Ошибка при загрузке ингредиентов");
        }
        const data: Ingredient[] = await response.json();
        setAvailableIngredients(data);
      } catch (err) {
        setError("Не удалось загрузить ингредиенты.");
      }
    };

    fetchIngredients();
  }, []);

  const handleSearch = async () => {
    if (ingredients.length === 0) {
      setError("Пожалуйста, выберите хотя бы один ингредиент.");
      return;
    }

    setLoading(true);
    setError(null);

    const ingredientNames = ingredients.map((ingredient) => ingredient.name);

    try {
      const response = await fetch("http://localhost:8000/api/recipes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ingredients: ingredientNames }),
      });

      if (!response.ok) {
        throw new Error("Ошибка при получении рецептов");
      }

      const data = await response.json();
      setRecipes(data);
    } catch (err) {
      setError("Не удалось загрузить рецепты. Попробуйте снова.");
    } finally {
      setLoading(false);
    }
  };

  // Обработчик для парсинга рецепта
  const handleParseRecipe = async (url: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/api/parse_recipe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error("Ошибка при парсинге рецепта");
      }

      const data = await response.json();
      navigate("/recipe-details", { state: { recipeDetails: data } }); // Переход на страницу с деталями
    } catch (err) {
      setError("Не удалось загрузить детали рецепта.");
    } finally {
      setLoading(false);
    }
  };

  // Обработчик для получения рекомендаций
  const handleRecommendations = async () => {
    if (!userId) {
      setError("Пожалуйста, введите ваш user_id.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/api/recommendations?user_id=${userId}`);
      if (!response.ok) {
        throw new Error("Ошибка при получении рекомендаций");
      }

      const data = await response.json();
      setRecommendedRecipes(data);
    } catch (err) {
      setError("Не удалось загрузить рекомендации.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container
      maxWidth="md"
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        py: 4,
      }}
    >
      <Typography
        variant="h3"
        component="h1"
        gutterBottom
        sx={{ fontWeight: "bold", mb: 4, textAlign: "center" }}
      >
        Найди рецепт по ингредиентам
      </Typography>

      {/* Поле для ввода user_id и кнопка "Рекомендация" */}
      <Box
        sx={{
          display: "flex",
          gap: 2,
          mb: 4,
          width: "100%",
          maxWidth: "600px",
          alignItems: "center",
        }}
      >
        <TextField
          label="Введите ваш user_id"
          variant="outlined"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          sx={{ width: "100%" }}
        />
        <Button
          variant="contained"
          onClick={handleRecommendations}
          sx={{ height: "56px", minWidth: "120px" }}
        >
          Рекомендация
        </Button>
      </Box>

      {/* Фильтр + кнопка */}
      <Box
        sx={{
          display: "flex",
          gap: 2,
          mb: 4,
          width: "100%",
          maxWidth: "600px",
          alignItems: "center",
        }}
      >
        <Autocomplete
          multiple
          id="ingredients"
          options={availableIngredients}
          value={ingredients}
          onChange={(_, newValue) => setIngredients(newValue)}
          disableCloseOnSelect
          getOptionLabel={(option) => option.name}
          renderInput={(params) => (
            <TextField
              {...params}
              label="Выберите ингредиенты"
              variant="outlined"
              sx={{ width: "100%" }}
            />
          )}
        />

        <Button
          variant="contained"
          onClick={handleSearch}
          sx={{ height: "56px", minWidth: "120px" }}
        >
          Найти
        </Button>
      </Box>

      {/* Выбранные ингредиенты */}
      {ingredients.length > 0 && (
        <Box
          sx={{
            display: "flex",
            flexWrap: "wrap",
            gap: 1,
            mt: 2,
            justifyContent: "center",
          }}
        >
          {ingredients.map((ingredient) => (
            <Chip key={ingredient.name} label={ingredient.name} color="primary" />
          ))}
        </Box>
      )}

      {/* Ошибка загрузки */}
      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}

      {/* Индикатор загрузки */}
      {loading && <CircularProgress sx={{ mt: 2 }} />}

      {/* Список рецептов */}
      <Grid container spacing={3} sx={{ width: "100%", mt: 4 }}>
        {recipes.length === 0 && !loading && (
          <Typography variant="body1" sx={{ mt: 2 }}>
            Рецепты не найдены. Попробуйте другие ингредиенты.
          </Typography>
        )}
        {recipes.map((recipe) => (
          <Grid item key={recipe.id} xs={12} sm={6} md={4}>
            <Card
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                boxShadow: 3,
                "&:hover": {
                  boxShadow: 6,
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="h6" component="h2" gutterBottom>
                  {recipe.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" component="div">
                  {recipe.ingredients.split(",").map((ingredient) => (
                    <Chip key={`${recipe.id}-${ingredient.trim()}`} label={ingredient.trim()} sx={{ m: 0.5 }} />
                  ))}
                </Typography>
                <Button
                  variant="contained"
                  onClick={() => handleParseRecipe(recipe.url)} // Обработчик для парсинга
                  sx={{ mt: 2 }}
                >
                  Подробнее
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Список рекомендованных рецептов */}
      {recommendedRecipes.length > 0 && (
        <Box sx={{ mt: 4, width: "100%" }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Рекомендованные рецепты
          </Typography>
          <Grid container spacing={3}>
            {recommendedRecipes.map((recipe) => (
              <Grid item key={recipe.id} xs={12} sm={6} md={4}>
                <Card
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    boxShadow: 3,
                    "&:hover": {
                      boxShadow: 6,
                    },
                  }}
                >
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" component="h2" gutterBottom>
                      {recipe.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" component="div">
                      {recipe.ingredients.split(",").map((ingredient) => (
                        <Chip key={`${recipe.id}-${ingredient.trim()}`} label={ingredient.trim()} sx={{ m: 0.5 }} />
                      ))}
                    </Typography>
                    <Button
                      variant="contained"
                      onClick={() => handleParseRecipe(recipe.url)} // Обработчик для парсинга
                      sx={{ mt: 2 }}
                    >
                      Подробнее
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Container>
  );
}