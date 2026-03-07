/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        brand: {
          50: "#eef9ff",
          100: "#d9f1ff",
          200: "#bce7ff",
          300: "#8ed9ff",
          400: "#59c2ff",
          500: "#33a6ff",
          600: "#1a87f5",
          700: "#136ee1",
          800: "#1658b6",
          900: "#184b8f",
          950: "#142f57",
        },
        surface: {
          50: "#f6f7f9",
          100: "#eceef2",
          200: "#d5dae2",
          300: "#b1bac9",
          400: "#8794ab",
          500: "#687690",
          600: "#535f76",
          700: "#444d61",
          800: "#3b4252",
          900: "#343946",
          950: "#22262e",
        },
      },
    },
  },
  plugins: [],
};
