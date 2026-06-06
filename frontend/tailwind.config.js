/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50:  '#f0f7f3',
          100: '#d8ede3',
          200: '#b3dcc6',
          300: '#83c4a5',
          400: '#52a882',
          500: '#338c65',
          600: '#267350',
          700: '#1d5c3f',
          800: '#164730',
          900: '#0f3321',
        },
      },
    },
  },
  plugins: [],
}
