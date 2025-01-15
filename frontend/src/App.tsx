import "./App.css";
import React, { useState } from "react";
import { Box } from "@mui/system";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Nav from "./components/Nav";

import CellImageGrid from "./components/CellOverview";
import Databases from "./components/Databases";
import TopPage from "./components/TopPage";
import Nd2Files from "./components/Nd2files";
import CellExtraction from "./components/CellExtraction";
import GraphEngine from "./components/GraphEngine";
import TimelapseNd2List from "./components/TimelapseNd2List";
import TimelapseParser from "./components/TimelapseParser";
import ResultsConsole from "./components/ResultsConsole";
import TimelapseDatabases from "./components/TimelapseDatabases";
// ↑ 必要なコンポーネントをインポート

/* --------------------------------
 *  PasswordProtect component
 * --------------------------------*/
const PasswordProtect: React.FC<{
  setIsAuthenticated: (isAuthenticated: boolean) => void;
}> = ({ setIsAuthenticated }) => {
  const [password, setPassword] = useState("");

  const correctPassword = "llama2";

  const handlePasswordSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (password === correctPassword) {
      setIsAuthenticated(true);
    } else {
      alert("Invalid password");
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
        bgcolor: "background.default",
      }}
    >
      <form onSubmit={handlePasswordSubmit}>
        <h2>Protected</h2>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="enter password"
          style={{ padding: "8px", margin: "8px", borderRadius: "4px" }}
        />
        <button type="submit" style={{ padding: "8px 16px" }}>
          authorize
        </button>
      </form>
    </Box>
  );
};

/* --------------------------------
 *  App component
 * --------------------------------*/
function App() {
  // パスワード認証の管理
  const [isAuthenticated, setIsAuthenticated] = useState(
    !(window.location.hostname === "10.32.17.73" && window.location.port === "3000")
  );

  return (
    <Box sx={{ bgcolor: "#fff", color: "black", minHeight: "100vh" }}>
      {isAuthenticated ? (
        <Router>
          {/* ルーティングの外側に Nav を配置し、常時表示しておく */}
          <Nav title="PhenoPixel5.0" />

          {/* ここからルーティング */}
          <Routes>
            <Route
              path="/"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <TopPage />
                </Box>
              }
            />
            <Route
              path="/dbconsole"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <Databases />
                </Box>
              }
            />
            <Route
              path="/databases"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <CellImageGrid />
                </Box>
              }
            />
            <Route
              path="/nd2files"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <Nd2Files />
                </Box>
              }
            />
            <Route
              path="/cellextraction"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <CellExtraction />
                </Box>
              }
            />
            <Route
              path="/graphengine"
              element={
                // Navが確保した余白をさらに詰めるなら p:1 など少なめに設定
                <Box component="main" sx={{ p: 1 }}>
                  <GraphEngine />
                </Box>
              }
            />
            <Route
              path="/tl-engine"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <TimelapseNd2List />
                </Box>
              }
            />
            <Route
              path="/tlparser"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <TimelapseParser />
                </Box>
              }
            />
            <Route
              path="/results"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <ResultsConsole />
                </Box>
              }
            />
            <Route
              path="/tlengine/dbconsole"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <TimelapseDatabases />
                </Box>
              }
            />
          </Routes>
        </Router>
      ) : (
        // パスワード保護
        <PasswordProtect setIsAuthenticated={setIsAuthenticated} />
      )}
    </Box>
  );
}

export default App;
