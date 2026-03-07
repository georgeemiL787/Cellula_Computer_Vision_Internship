import { useCallback, useState } from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import type { ProcessedResult } from "./types";
import { Layout } from "./components/Layout";
import { Gallery } from "./components/Gallery";
import { DetailView } from "./components/DetailView";
import { UploadFlow } from "./components/UploadFlow";

function App() {
  const [results, setResults] = useState<ProcessedResult[]>([]);
  const navigate = useNavigate();

  const addResult = useCallback((result: ProcessedResult) => {
    setResults((prev) => [result, ...prev]);
    navigate(`/detail/${result.id}`);
  }, [navigate]);

  const removeResult = useCallback((id: string) => {
    setResults((prev) => prev.filter((r) => r.id !== id));
    navigate("/");
  }, [navigate]);

  const getResult = useCallback((id: string) => results.find((r) => r.id === id), [results]);

  return (
    <Layout>
      <Routes>
        <Route
          path="/"
          element={
            <>
              <UploadFlow onComplete={addResult} />
              <Gallery results={results} onRemove={removeResult} />
            </>
          }
        />
        <Route
          path="/detail/:id"
          element={
            <DetailView getResult={getResult} onRemove={removeResult} onBack={() => navigate("/")} />
          }
        />
      </Routes>
    </Layout>
  );
}

export default App;
