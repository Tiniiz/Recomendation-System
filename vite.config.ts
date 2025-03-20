import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    react({
      jsxImportSource: '@emotion/react', // Добавьте эту строку
      babel: {
        plugins: ['@emotion/babel-plugin'], // Добавьте эту строку
      },
    }),
  ],
});