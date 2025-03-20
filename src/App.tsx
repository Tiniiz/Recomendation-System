import RecipeFinder from "./RecipeFinder";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import RecipeDetails from "./RecipeDetails";
import { createTheme, ThemeProvider } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2", // Основной цвет
    },
  },
  typography: {
    fontFamily: "Roboto, sans-serif",
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
      <Routes>
        <Route path="/" element={<RecipeFinder />} />
        <Route path="/recipe-details" element={<RecipeDetails />} />
      </Routes>
    </Router>
    </ThemeProvider>
  );
}

export default App;