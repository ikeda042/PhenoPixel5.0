import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Container,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { settings } from "../settings";

const { url_prefix } = settings;

interface UserAccount {
  id: string;
  handle_id: string;
  scopes: string[];
}

const UserInfo: React.FC = () => {
  const [userInfo, setUserInfo] = useState<UserAccount | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const navigate = useNavigate();

  const fetchUserInfo = async () => {
    setLoading(true);
    setError("");
    const accessToken = localStorage.getItem("access_token");
    if (!accessToken) {
      setError("No access token found. Please login.");
      navigate("/login");
      return;
    }
    try {
      const response = await fetch(`${url_prefix}/oauth2/me`, {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${accessToken}`,
        },
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to fetch user info");
      }

      const data = await response.json();
      setUserInfo(data.account);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUserInfo();
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    navigate("/login");
  };

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          mt: 8,
        }}
      >
        <Typography variant="h4" component="h2" gutterBottom>
          User Information
        </Typography>
        {error && (
          <Alert severity="error" sx={{ width: "100%", mb: 2 }}>
            {error}
          </Alert>
        )}
        {loading ? (
          <CircularProgress />
        ) : userInfo ? (
          <Box sx={{ width: "100%", textAlign: "left", mb: 2 }}>
            <Typography variant="body1">
              <strong>ID:</strong> {userInfo.id}
            </Typography>
            <Typography variant="body1">
              <strong>Username:</strong> {userInfo.handle_id}
            </Typography>
            <Typography variant="body1">
              <strong>Scopes:</strong> {userInfo.scopes.join(", ")}
            </Typography>
          </Box>
        ) : (
          <Typography variant="body1">No user info available</Typography>
        )}
        <Box sx={{ display: "flex", gap: 2 }}>
          <Button variant="contained" color="primary" onClick={fetchUserInfo}>
            Refresh
          </Button>
          <Button variant="outlined" color="secondary" onClick={handleLogout}>
            Logout
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default UserInfo;
