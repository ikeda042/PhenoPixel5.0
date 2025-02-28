import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Box, TextField, Button } from "@mui/material";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // Build URL-encoded form data
    const formData = new URLSearchParams();
    formData.append("grant_type", "password");
    formData.append("username", username);
    formData.append("password", password);
    // OAuth2RequestForm expects "scope" as a string, space separated.
    formData.append("scope", "me");

    try {
      const response = await fetch(`${url_prefix}/oauth2/token`, {
        method: "POST",
        headers: {
          "Origin": "http://localhost:3000",  
          // Do not set "Content-Type" header; fetch will set it automatically for URLSearchParams
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
      navigate("/user_info");
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
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        mt: 8,
      }}
    >
      <h2>Login</h2>
      <form onSubmit={handleSubmit}>
        <TextField
          label="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          margin="normal"
          fullWidth
        />
        <TextField
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          margin="normal"
          fullWidth
        />
        {error && <p style={{ color: "red" }}>{error}</p>}
        <Button type="submit" variant="contained" color="primary" sx={{ mt: 2 }}>
          Login
        </Button>
      </form>
    </Box>
  );
};

export default Login;
