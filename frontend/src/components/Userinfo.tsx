import React, { useState, useEffect } from "react";
import {
  Box,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Container,
  TextField,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
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
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const [oldPassword, setOldPassword] = useState<string>("");
  const [newPassword, setNewPassword] = useState<string>("");
  const [passwordChangeError, setPasswordChangeError] = useState<string>("");
  const [passwordChangeSuccess, setPasswordChangeSuccess] = useState<string>("");
  const [passwordChangeLoading, setPasswordChangeLoading] = useState<boolean>(false);

  // プルダウンで選択するメニュー（"info"：ユーザー情報, "password"：パスワード変更）
  const [selectedMenu, setSelectedMenu] = useState<"info" | "password">("info");

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
          Authorization: `Bearer ${accessToken}`,
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

  const handleChangePassword = async () => {
    setPasswordChangeError("");
    setPasswordChangeSuccess("");
    setPasswordChangeLoading(true);
    const accessToken = localStorage.getItem("access_token");
    if (!accessToken) {
      setPasswordChangeError("No access token found. Please login.");
      setPasswordChangeLoading(false);
      return;
    }
    try {
      const response = await fetch(`${url_prefix}/oauth2/change_password`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({
          old_password: oldPassword,
          new_password: newPassword,
        }),
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to change password");
      }
      const data = await response.json();
      setPasswordChangeSuccess("Password updated successfully.");
      // 更新されたユーザー情報で画面を更新
      setUserInfo(data.account);
      // 入力フィールドをクリア
      setOldPassword("");
      setNewPassword("");
    } catch (err: unknown) {
      if (err instanceof Error) {
        setPasswordChangeError(err.message);
      } else {
        setPasswordChangeError("An unexpected error occurred");
      }
    } finally {
      setPasswordChangeLoading(false);
    }
  };

  // プルダウンの選択変更
  const handleMenuChange = (event: SelectChangeEvent<"info" | "password">) => {
    setSelectedMenu(event.target.value as "info" | "password");
  };

  return (
    <Container maxWidth="sm">
      <Paper elevation={3} sx={{ p: 3 }}>
        {/* プルダウンメニュー */}
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel id="select-label">メニュー</InputLabel>
          <Select
            labelId="select-label"
            value={selectedMenu}
            onChange={handleMenuChange}
            label="メニュー"
          >
            <MenuItem value="info">ユーザー情報</MenuItem>
            <MenuItem value="password">パスワード変更</MenuItem>
          </Select>
        </FormControl>

        {selectedMenu === "info" && (
          <>
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
              <Typography variant="body1">ユーザー情報がありません</Typography>
            )}
            <Box sx={{ display: "flex", gap: 2, mb: 4 }}>
              <Button variant="contained" color="primary" onClick={fetchUserInfo}>
                更新
              </Button>
              <Button variant="outlined" color="secondary" onClick={handleLogout}>
                ログアウト
              </Button>
            </Box>
          </>
        )}

        {selectedMenu === "password" && (
          <>
            {passwordChangeError && (
              <Alert severity="error" sx={{ width: "100%", mb: 2 }}>
                {passwordChangeError}
              </Alert>
            )}
            {passwordChangeSuccess && (
              <Alert severity="success" sx={{ width: "100%", mb: 2 }}>
                {passwordChangeSuccess}
              </Alert>
            )}
            <TextField
              label="現在のパスワード"
              type="password"
              variant="outlined"
              fullWidth
              margin="normal"
              value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)}
            />
            <TextField
              label="新しいパスワード"
              type="password"
              variant="outlined"
              fullWidth
              margin="normal"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={handleChangePassword}
              disabled={passwordChangeLoading}
              sx={{ mt: 2 }}
            >
              {passwordChangeLoading ? <CircularProgress size={24} /> : "パスワード更新"}
            </Button>
          </>
        )}
      </Paper>
    </Container>
  );
};

export default UserInfo;
