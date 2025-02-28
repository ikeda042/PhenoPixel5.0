import React, { useState, useEffect } from "react";
import { Box, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

interface UserAccount {
  id: string;
  handle_id: string;
  scopes: string[];
}

const UserInfo: React.FC = () => {
  const [userInfo, setUserInfo] = useState<UserAccount | null>(null);
  const [error, setError] = useState<string>("");
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserInfo = async () => {
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
      }
    };

    fetchUserInfo();
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    navigate("/login");
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", mt: 8 }}>
      <h2>User Information</h2>
      {error && <p style={{ color: "red" }}>{error}</p>}
      {userInfo ? (
        <Box>
          <p>ID: {userInfo.id}</p>
          <p>Handle: {userInfo.handle_id}</p>
          <p>Scopes: {userInfo.scopes.join(", ")}</p>
        </Box>
      ) : (
        <p>Loading...</p>
      )}
      <Button variant="contained" onClick={handleLogout} sx={{ mt: 2 }}>
        Logout
      </Button>
    </Box>
  );
};

export default UserInfo;
