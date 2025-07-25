import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Box, TextField, Button, Paper, Typography } from "@mui/material";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [isRegister, setIsRegister] = useState(false);
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
    setError("");
    if (isRegister) {
      // ユーザー登録処理
      try {
        const response = await fetch(`${url_prefix}/oauth2/register`, {
          method: "POST",
          headers: {
            "Accept": "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            handle_id: username,
            password: password,
          }),
        });
        if (!response.ok) {
          const errData = await response.json();
          throw new Error(errData.detail || "Registration failed");
        }
        alert("User registered successfully. Please log in.");
        setIsRegister(false);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("An unexpected error occurred");
        }
      }
    } else {
      // ログイン処理
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
          // "Failed to fetch" の場合はエラーメッセージを上書きする
          if (err.message === "Failed to fetch") {
            setError("Invalid password or user name");
          } else {
            setError(err.message);
          }
        } else {
          setError("An unexpected error occurred");
        }
      }
    }
  };

  const toggleMode = () => {
    setError("");
    setIsRegister(!isRegister);
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
        <Typography variant="h4" sx={{ mb: 3, textAlign: "center", color: 'text.primary' }}>
          {isRegister ? "Register" : "Login"}
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
              backgroundColor: 'background.paper',
              input: { color: 'text.primary' },
              '& .MuiInputLabel-root': { color: 'text.primary' },
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'text.primary' },
                '&:hover fieldset': { borderColor: 'text.primary' },
                '&.Mui-focused fieldset': { borderColor: 'text.primary' },
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
              backgroundColor: 'background.paper',
              input: { color: 'text.primary' },
              '& .MuiInputLabel-root': { color: 'text.primary' },
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'text.primary' },
                '&:hover fieldset': { borderColor: 'text.primary' },
                '&.Mui-focused fieldset': { borderColor: 'text.primary' },
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
              backgroundColor: 'background.paper',
              color: 'text.primary',
              border: '1px solid',
              borderColor: 'text.primary',
              '&:hover': { backgroundColor: 'action.hover' },
            }}
          >
            {isRegister ? "Register" : "Login"}
          </Button>
        </form>
        <Button onClick={toggleMode} fullWidth sx={{ mt: 2 }} variant="text">
          {isRegister ? "Already have an account? Login" : "Don't have an account? Register"}
        </Button>
      </Paper>
    </Box>
  );
};

export default Login;
