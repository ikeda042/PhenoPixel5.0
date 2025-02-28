import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Box, TextField, Button, Paper, Typography } from "@mui/material";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  // 既にログインしている場合は "/" にリダイレクト
  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (token) {
      navigate("/", { replace: true });
    }
  }, [navigate]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new URLSearchParams();
    formData.append("grant_type", "password");
    formData.append("username", username);
    formData.append("password", password);
    formData.append("scope", "me");

    try {
      const response = await fetch(`${url_prefix}/oauth2/token`, {
        method: "POST",
        headers: {
          "Origin": "http://localhost:3000",
        },
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Login failed");
      }

      const data = await response.json();
      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("refresh_token", data.refresh_token);
      navigate("/", { replace: true });
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred");
      }
    }
  };

  return (
    <Box
      sx={{
        backgroundColor: "#fff",
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        p: 2,
      }}
    >
      <Paper
        elevation={3}
        sx={{
          padding: 4,
          maxWidth: 400,
          width: "100%",
          backgroundColor: "#fff",
          borderRadius: 2,
        }}
      >
        <Typography variant="h4" sx={{ mb: 3, textAlign: "center", color: "#000" }}>
          Login
        </Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            label="Username"
            variant="outlined"
            fullWidth
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            margin="normal"
            sx={{
              backgroundColor: "#fff",
              input: { color: "#000" },
              "& .MuiInputLabel-root": { color: "#000" },
              "& .MuiOutlinedInput-root": {
                "& fieldset": { borderColor: "#000" },
                "&:hover fieldset": { borderColor: "#000" },
                "&.Mui-focused fieldset": { borderColor: "#000" },
              },
            }}
          />
          <TextField
            label="Password"
            variant="outlined"
            type="password"
            fullWidth
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            margin="normal"
            sx={{
              backgroundColor: "#fff",
              input: { color: "#000" },
              "& .MuiInputLabel-root": { color: "#000" },
              "& .MuiOutlinedInput-root": {
                "& fieldset": { borderColor: "#000" },
                "&:hover fieldset": { borderColor: "#000" },
                "&.Mui-focused fieldset": { borderColor: "#000" },
              },
            }}
          />
          {error && (
            <Typography variant="body2" color="error" sx={{ mt: 1 }}>
              {error}
            </Typography>
          )}
          <Button
            type="submit"
            variant="contained"
            fullWidth
            sx={{
              mt: 3,
              backgroundColor: "#fff",
              color: "#000",
              border: "1px solid #000",
              "&:hover": { backgroundColor: "#f0f0f0" },
            }}
          >
            Login
          </Button>
        </form>
      </Paper>
    </Box>
  );
};

export default Login;
