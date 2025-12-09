# 🎬 Movie Recommender — LightFM + FastAPI + Docker + Netlify

A fully working **hybrid recommendation system** built with **LightFM embeddings**, served through a **FastAPI backend**, containerized using **Docker**, and exposed through a modern **React (Vite) frontend deployed on Netlify**.

The system generates personalized movie recommendations using **latent collaborative filtering embeddings** trained on the MovieLens dataset.

The project includes **end-to-end architecture**: data ingestion, model training, persistence, inference API, containerized backend, and cloud UI hosting.

---

## 🚀 Live Demo

### Frontend (Netlify)
👉 https://incomparable-ganache-b1dc03.netlify.app/

### Backend REST API (local Docker)
👉 http://localhost:8000/docs

---

## 🔥 Architecture Overview

| Layer | Tool |
|------|------|
| Frontend | React (Vite, Netlify) |
| Backend | FastAPI |
| ML Model | LightFM Hybrid Recommender |
| Infra | Docker Compose |
| Database | PostgreSQL |

```
frontend/
backend/
infra/
scripts/
```

---

## 🧠 Why LightFM?

LightFM combines:

- collaborative filtering  
- content-based filtering  
- latent factor embeddings  
- user + item feature support  

### Benefits
✔ handles sparse data extremely well  
✔ learns meaningful latent representations  
✔ top performance on MovieLens  
✔ production-ready embeddings  

---

# ⚙️ FastAPI Backend

## Example endpoint
```
GET /recommendations/by-title?title=Spider%20man&top_k=5
```

## Example response
```json
[
  {"movie_id": 6333, "title": "X2: X-Men United (2003)", "score": 0.97},
  ...
]
```

---

# 🐳 Docker — FastAPI Backend

### Public Docker Hub
https://hub.docker.com/r/mher1902/movie-backend

### Pull image
```bash
docker pull mher1902/movie-backend:latest
```

### Run locally
```bash
docker run -p 8000:8000 mher1902/movie-backend:latest
```

---

## 🧩 Model Artifacts

```
backend/app/recommender/artifacts/
  ├── lightfm_model.pkl
  ├── user_id_map.pkl
  └── item_id_map.pkl
```

Recommender uses cosine similarity over latent embeddings.

---

## 🐳 Docker + Local Infra

Local services:
- FastAPI backend  
- PostgreSQL database  
- (future) Cloud SQL / Render / Railway deployment  

### Run locally
```bash
cd infra
docker compose -f docker-compose.local.yml up -d
```

---

## 🌐 Frontend — React + Vite + Netlify

### Features
- modern UI  
- movie search  
- recommendation results  
- API integration  
- global CDN hosting  

### Build
```bash
cd frontend
npm install
npm run build
```

### Local development
```bash
npm run dev
```

---

## 🌍 Deployment

### Frontend
Netlify  
- free  
- automatic deploys  
- HTTPS  
- CDN  

### Backend (future)
- Render  
- Railway  
- Cloud Run  
- Nginx reverse proxy  

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| ML | LightFM, numpy, scipy |
| API | FastAPI, Pydantic |
| DB | PostgreSQL |
| Infra | Docker, Docker Compose |
| Frontend | React, Vite |
| Deploy | Netlify |
| Data | MovieLens |
| Languages | Python, JavaScript |

---

## 🎯 Why These Choices?

| Decision | Reason |
|---------|--------|
| LightFM | strong hybrid recommender |
| FastAPI | async high-performance backend |
| Docker | portable infra |
| Netlify | simple global hosting |
| PostgreSQL | reliable relational DB |
| Vite | modern React tooling |

---

## 💻 Local Development

### Backend + DB
```bash
cd infra
docker compose -f docker-compose.local.yml up -d
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### API docs
http://localhost:8000/docs

### UI
http://localhost:5173

---

## 🚧 Next Steps
- deploy backend to cloud  
- expose public API  
- connect UI to public URL  
- integrate genre features  
- add user login & personalization  

---

## 👤 Author
**Mher Sukiasyan**  
Machine Learning & Data Engineering  

GitHub: https://github.com/Mher-1902  

---

## ⭐ Inspiration
MovieLens dataset + LightFM research + production design patterns.
