import { useState } from "react";
import { getRecommendationsByTitle } from "./api";
import "./App.css";

function App() {
  const [title, setTitle] = useState("");
  const [topK, setTopK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults([]);

    if (!title.trim()) {
      setError("Please enter a movie title.");
      return;
    }

    try {
      setLoading(true);
      const data = await getRecommendationsByTitle(title.trim(), topK);
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="logo-circle">MR</div>
        <div className="header-text">
          <h1>Movie Recommender</h1>
          <p>Personalized suggestions powered by your LightFM model.</p>
        </div>
      </header>

      <main className="app-main">
        <section className="hero">
          <div className="hero-text">
            <h2>Find your next favorite movie.</h2>
            <p>
              Type a movie you like and we&apos;ll recommend similar titles based
              on collaborative filtering and latent embeddings.
            </p>
            <form className="search-form" onSubmit={handleSubmit}>
              <div className="input-row">
                <input
                  type="text"
                  placeholder="Search by title… (e.g. Spider man)"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  title="Number of recommendations"
                />
                <button type="submit" disabled={loading}>
                  {loading ? "Searching…" : "Recommend"}
                </button>
              </div>
              {error && <div className="error-banner">{error}</div>}
              {!error && !loading && results.length === 0 && (
                <p className="hint-text">
                  Try a popular title like <span>Spider man</span> or{" "}
                  <span>Matrix</span>.
                </p>
              )}
            </form>
          </div>
        </section>

        <section className="results-section">
          <div className="results-header">
            <h3>Recommendations</h3>
            {results.length > 0 && (
              <span className="results-count">
                {results.length} result{results.length > 1 ? "s" : ""}
              </span>
            )}
          </div>

          {loading && (
            <div className="skeleton-grid">
              {Array.from({ length: 6 }).map((_, i) => (
                <div className="skeleton-card" key={i} />
              ))}
            </div>
          )}

          {!loading && results.length > 0 && (
            <div className="card-grid">
              {results.map((movie, idx) => (
                <article className="movie-card" key={movie.movie_id ?? idx}>
                  <div className="card-header">
                    <div className="poster-placeholder">
                      {movie.title?.[0] ?? "M"}
                    </div>
                    <div>
                      <h4>{movie.title}</h4>
                      <p className="movie-id">ID: {movie.movie_id}</p>
                    </div>
                  </div>
                  <div className="card-footer">
                    <span className="score-label">relevance</span>
                    <span className="score-value">
                      {movie.score != null
                        ? movie.score.toFixed(3)
                        : "N/A"}
                    </span>
                  </div>
                </article>
              ))}
            </div>
          )}

          {!loading && !error && results.length === 0 && (
            <div className="empty-state">
              <p>No recommendations yet. Start by searching for a movie.</p>
            </div>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Built with <span>FastAPI</span>, <span>LightFM</span> and{" "}
          <span>React</span>.
        </p>
      </footer>
    </div>
  );
}

export default App;
