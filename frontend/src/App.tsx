import "./App.css";
import React, { useState, useMemo, useEffect } from "react";
import { Box } from "@mui/system";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
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
import TimelapseViewer from "./components/TimelapseCellOverview";
import Login from "./components/Login";
import UserInfo from "./components/Userinfo";
import LabelSorter from "./components/LabelSorter";
import CDT from "./components/CDT";
import MiniFileManager from "./components/MiniFileManager";
import ImagePlayground from "./components/ImagePlayground";
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
   true
  );

  const [mode, setMode] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('themeMode');
    if (saved === 'light' || saved === 'dark') {
      return saved as 'light' | 'dark';
    }
    return 'dark';
  });

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          ...(mode === "dark"
            ? {
                primary: {
                  main: "#ffffff",
                  dark: "#cccccc",
                },
              }
            : {
                primary: {
                  main: "#000000",
                  dark: "#333333",
                  contrastText: "#ffffff",
                },
              }),
        },
      }),
    [mode]
  );

  useEffect(() => {
    localStorage.setItem('themeMode', mode);
  }, [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ bgcolor: 'background.default', color: 'text.primary', minHeight: '100vh' }}>
      {isAuthenticated ? (
        <Router>
          {/* ルーティングの外側に Nav を配置し、常時表示しておく */}
          <Nav title="PhenoPixel5.0" mode={mode} toggleMode={() => setMode(prev => prev === 'light' ? 'dark' : 'light')} />

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
              path="/labelsorter"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <LabelSorter />
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
              path="/cdt"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <CDT />
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
            <Route
              path="/tlengine/databases/"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <TimelapseViewer />
                </Box>
              }
            />
            <Route
              path="/files"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <MiniFileManager />
                </Box>
              }
            />
            <Route
              path="/image-playground"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <ImagePlayground />
                </Box>
              }
            />
            <Route
              path="/login"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <Login />
                </Box>
              }
            />
            <Route
              path="/user_info"
              element={
                <Box component="main" sx={{ p: 1 }}>
                  <UserInfo />
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
    </ThemeProvider>
  );
}

export default App;
