/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'hospital-blue': '#0ea5e9',
        'hospital-light-blue': '#e0f2fe',
        'hospital-dark-blue': '#0284c7',
      },
    },
  },
  plugins: [],
}

