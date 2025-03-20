import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Typography, Box, Button, CircularProgress } from "@mui/material";

interface RecipeDetails {
  name: string;
  details: string;
  cook_time: string;
  nutrition_data: string;
  steps: string;
}

export default function RecipeDetails() {
  const location = useLocation();
  const navigate = useNavigate();
  const [recipeDetails, setRecipeDetails] = useState<RecipeDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (location.state?.recipeDetails) {
      setRecipeDetails(location.state.recipeDetails);
    } else {
      navigate("/"); // Если данные отсутствуют, вернуться на главную
    }
  }, [location, navigate]);

  if (!recipeDetails) {
    return <CircularProgress />;
  }

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        {recipeDetails.name}
      </Typography>
      <Typography variant="body1" gutterBottom>
        <strong>Время приготовления:</strong> {recipeDetails.cook_time}
      </Typography>
      <Typography variant="body1" gutterBottom>
        <strong>Ингредиенты:</strong> {recipeDetails.details}
      </Typography>
      <Typography variant="body1" gutterBottom>
        <strong>Пищевая ценность:</strong> {recipeDetails.nutrition_data}
      </Typography>
      <Typography variant="body1" gutterBottom>
        <strong>Шаги приготовления:</strong> {recipeDetails.steps}
      </Typography>
      <Button variant="contained" onClick={() => navigate("/")} sx={{ mt: 2 }}>
        Назад
      </Button>
    </Box>
  );
}