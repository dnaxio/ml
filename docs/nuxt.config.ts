export default defineNuxtConfig({
  extends: ["@baybreezy/docd"],
  llms: {
    domain: process.env.NUXT_SITE_URL || "http://localhost:3000",
    title: "Dnax ML — Lightweight Machine Learning for Devs & Humans",
    description:
      "JSON-first machine learning in TypeScript: train and predict directly on arrays of row objects — no matrix conversion, no Python runtime.",
    full: {
      title: "Dnax ML — Lightweight Machine Learning for Devs & Humans",
      description:
        "JSON-first machine learning in TypeScript: train and predict directly on arrays of row objects — no matrix conversion, no Python runtime.",
    },
  },
});
