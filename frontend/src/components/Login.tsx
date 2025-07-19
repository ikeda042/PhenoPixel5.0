import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { 
  Box, 
  TextField, 
  Button, 
  Paper, 
  Typography, 
  Container,
  Avatar,
  Chip,
  Fade,
  Slide,
  IconButton,
  InputAdornment
} from "@mui/material";
import { 
  Lock as LockIcon, 
  Person as PersonIcon,
  Visibility,
  VisibilityOff,
  LoginOutlined,
  PersonAddOutlined
} from "@mui/icons-material";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

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
    setIsLoading(true);

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
    setIsLoading(false);
  };

  const toggleMode = () => {
    setError("");
    setIsRegister(!isRegister);
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <Box
      sx={{
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        position: "relative",
        overflow: "hidden",
        "&::before": {
          content: '""',
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><defs><pattern id=\"grain\" width=\"100\" height=\"100\" patternUnits=\"userSpaceOnUse\"><circle cx=\"50\" cy=\"50\" r=\"0.5\" fill=\"%23ffffff\" opacity=\"0.1\"/></pattern></defs><rect width=\"100\" height=\"100\" fill=\"url(%23grain)\"/></svg>')",
          opacity: 0.3,
        },
      }}
    >
      {/* Floating background elements */}
      <Box
        sx={{
          position: "absolute",
          top: "10%",
          left: "10%",
          width: "200px",
          height: "200px",
          background: "rgba(255, 255, 255, 0.1)",
          borderRadius: "50%",
          animation: "float 6s ease-in-out infinite",
          "@keyframes float": {
            "0%, 100%": { transform: "translateY(0px) rotate(0deg)" },
            "50%": { transform: "translateY(-20px) rotate(180deg)" },
          },
        }}
      />
      <Box
        sx={{
          position: "absolute",
          bottom: "15%",
          right: "15%",
          width: "150px",
          height: "150px",
          background: "rgba(255, 255, 255, 0.08)",
          borderRadius: "50%",
          animation: "float 8s ease-in-out infinite reverse",
        }}
      />

      <Container maxWidth="sm">
        <Fade in timeout={1000}>
          <Paper
            elevation={24}
            sx={{
              padding: { xs: 3, sm: 4, md: 5 },
              background: "rgba(255, 255, 255, 0.95)",
              backdropFilter: "blur(20px)",
              borderRadius: "24px",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1)",
              position: "relative",
              overflow: "hidden",
              "&::before": {
                content: '""',
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "4px",
                background: "linear-gradient(90deg, #667eea, #764ba2, #667eea)",
                backgroundSize: "200% 100%",
                animation: "shimmer 2s linear infinite",
                "@keyframes shimmer": {
                  "0%": { backgroundPosition: "200% 0" },
                  "100%": { backgroundPosition: "-200% 0" },
                },
              },
            }}
          >
            {/* Header Section */}
            <Box sx={{ textAlign: "center", mb: 4 }}>
              <Slide direction="down" in timeout={800}>
                <Avatar
                  sx={{
                    width: 80,
                    height: 80,
                    margin: "0 auto 16px",
                    background: "linear-gradient(135deg, #667eea, #764ba2)",
                    fontSize: "2rem",
                  }}
                >
                  {isRegister ? <PersonAddOutlined /> : <LoginOutlined />}
                </Avatar>
              </Slide>
              
              <Typography
                variant="h3"
                sx={{
                  mb: 1,
                  background: "linear-gradient(135deg, #667eea, #764ba2)",
                  backgroundClip: "text",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  fontWeight: 700,
                  fontSize: { xs: "2rem", sm: "2.5rem" },
                }}
              >
                {isRegister ? "Create Account" : "Welcome Back"}
              </Typography>
              
              <Typography
                variant="subtitle1"
                sx={{
                  color: "text.secondary",
                  mb: 2,
                  opacity: 0.8,
                }}
              >
                {isRegister ? "Join us today and get started" : "Sign in to continue to your account"}
              </Typography>

              {/* Mode Toggle Chips */}
              <Box sx={{ display: "flex", justifyContent: "center", gap: 1, mb: 3 }}>
                <Chip
                  label="Login"
                  onClick={() => !isRegister || toggleMode()}
                  sx={{
                    background: !isRegister 
                      ? "linear-gradient(135deg, #667eea, #764ba2)" 
                      : "transparent",
                    color: !isRegister ? "white" : "text.secondary",
                    fontWeight: 600,
                    "&:hover": { 
                      background: !isRegister 
                        ? "linear-gradient(135deg, #667eea, #764ba2)" 
                        : "rgba(102, 126, 234, 0.1)" 
                    },
                  }}
                />
                <Chip
                  label="Register"
                  onClick={() => isRegister || toggleMode()}
                  sx={{
                    background: isRegister 
                      ? "linear-gradient(135deg, #667eea, #764ba2)" 
                      : "transparent",
                    color: isRegister ? "white" : "text.secondary",
                    fontWeight: 600,
                    "&:hover": { 
                      background: isRegister 
                        ? "linear-gradient(135deg, #667eea, #764ba2)" 
                        : "rgba(102, 126, 234, 0.1)" 
                    },
                  }}
                />
              </Box>
            </Box>

            {/* Form Section */}
            <Slide direction="up" in timeout={1000}>
              <form onSubmit={handleSubmit}>
                <TextField
                  label="Username"
                  variant="outlined"
                  fullWidth
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  margin="normal"
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <PersonIcon sx={{ color: "text.secondary" }} />
                      </InputAdornment>
                    ),
                  }}
                  sx={{
                    mb: 2,
                    "& .MuiOutlinedInput-root": {
                      borderRadius: "12px",
                      backgroundColor: "rgba(255, 255, 255, 0.8)",
                      transition: "all 0.3s ease",
                      "&:hover": {
                        backgroundColor: "rgba(255, 255, 255, 0.9)",
                        transform: "translateY(-2px)",
                        boxShadow: "0 4px 20px rgba(102, 126, 234, 0.1)",
                      },
                      "&.Mui-focused": {
                        backgroundColor: "white",
                        transform: "translateY(-2px)",
                        boxShadow: "0 4px 20px rgba(102, 126, 234, 0.2)",
                        "& fieldset": {
                          borderColor: "#667eea",
                          borderWidth: "2px",
                        },
                      },
                    },
                  }}
                />

                <TextField
                  label="Password"
                  variant="outlined"
                  type={showPassword ? "text" : "password"}
                  fullWidth
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  margin="normal"
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <LockIcon sx={{ color: "text.secondary" }} />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          onClick={togglePasswordVisibility}
                          edge="end"
                          sx={{ color: "text.secondary" }}
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                  sx={{
                    mb: 3,
                    "& .MuiOutlinedInput-root": {
                      borderRadius: "12px",
                      backgroundColor: "rgba(255, 255, 255, 0.8)",
                      transition: "all 0.3s ease",
                      "&:hover": {
                        backgroundColor: "rgba(255, 255, 255, 0.9)",
                        transform: "translateY(-2px)",
                        boxShadow: "0 4px 20px rgba(102, 126, 234, 0.1)",
                      },
                      "&.Mui-focused": {
                        backgroundColor: "white",
                        transform: "translateY(-2px)",
                        boxShadow: "0 4px 20px rgba(102, 126, 234, 0.2)",
                        "& fieldset": {
                          borderColor: "#667eea",
                          borderWidth: "2px",
                        },
                      },
                    },
                  }}
                />

                {error && (
                  <Fade in timeout={300}>
                    <Paper
                      sx={{
                        p: 2,
                        mb: 3,
                        background: "rgba(244, 67, 54, 0.1)",
                        border: "1px solid rgba(244, 67, 54, 0.3)",
                        borderRadius: "12px",
                      }}
                    >
                      <Typography variant="body2" color="error" sx={{ fontWeight: 500 }}>
                        {error}
                      </Typography>
                    </Paper>
                  </Fade>
                )}

                <Button
                  type="submit"
                  variant="contained"
                  fullWidth
                  disabled={isLoading}
                  sx={{
                    mt: 2,
                    mb: 2,
                    py: 1.5,
                    borderRadius: "12px",
                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    fontSize: "1.1rem",
                    fontWeight: 600,
                    textTransform: "none",
                    boxShadow: "0 4px 15px rgba(102, 126, 234, 0.4)",
                    transition: "all 0.3s ease",
                    "&:hover": {
                      background: "linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)",
                      transform: "translateY(-3px)",
                      boxShadow: "0 6px 25px rgba(102, 126, 234, 0.6)",
                    },
                    "&:active": {
                      transform: "translateY(-1px)",
                    },
                    "&:disabled": {
                      background: "rgba(0, 0, 0, 0.12)",
                      color: "rgba(0, 0, 0, 0.26)",
                      boxShadow: "none",
                      transform: "none",
                    },
                  }}
                >
                  {isLoading ? "Processing..." : (isRegister ? "Create Account" : "Sign In")}
                </Button>
              </form>
            </Slide>

            {/* Footer */}
            <Box sx={{ textAlign: "center", pt: 2 }}>
              <Typography variant="body2" sx={{ color: "text.secondary", mb: 1 }}>
                {isRegister ? "Already have an account?" : "Don't have an account?"}
              </Typography>
              <Button
                onClick={toggleMode}
                variant="text"
                sx={{
                  textTransform: "none",
                  fontWeight: 600,
                  color: "#667eea",
                  "&:hover": {
                    background: "rgba(102, 126, 234, 0.1)",
                  },
                }}
              >
                {isRegister ? "Sign in instead" : "Create one now"}
              </Button>
            </Box>
          </Paper>
        </Fade>
      </Container>
    </Box>
  );
};

export default Login;