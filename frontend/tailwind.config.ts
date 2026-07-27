import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      boxShadow: { soft: "0 12px 32px rgba(15, 23, 42, 0.08)" },
    },
  },
  plugins: [],
} satisfies Config;
