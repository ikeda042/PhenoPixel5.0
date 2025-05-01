// src/components/GraphEngine.tsx
import React, { useState } from "react";
import {
  Box,
  Container,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  CircularProgress,
  Breadcrumbs,
  Link,
  Paper,
  useMediaQuery,
  TextField,
  Stack,
  Divider,
} from "@mui/material";
import { styled, useTheme } from "@mui/material/styles";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

/* -------------------------------------------------
 * UI helpers
 * ------------------------------------------------- */

// “黒ベース”のボタン
const BlackButton = styled("button")(({ theme }) => ({
  // reset default button styles
  appearance: "none",
  border: "none",
  outline: "none",
  padding: 0,
  margin: 0,
  font: "inherit",

  /* size & layout */
  width: "100%",
  minHeight: 40,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: theme.spacing(1),
  borderRadius: theme.shape.borderRadius,

  /* colors */
  backgroundColor: "#000",
  color: "#fff",

  /* typography */
  fontWeight: 500,
  letterSpacing: 0.5,
  textDecoration: "none",
  cursor: "pointer",
  textTransform: "none",

  /* interaction */
  transition: "background-color 0.2s ease",

  "&:hover": {
    backgroundColor: "#222",
  },
  "&:disabled": {
    backgroundColor: theme.palette.action.disabledBackground,
    color: theme.palette.action.disabled,
    cursor: "not-allowed",
  },
}));

/* -------------------------------------------------
 * Component
 * ------------------------------------------------- */

const GraphEngine: React.FC = () => {
  const [mode, setMode] = useState<"heatmap_abs" | "heatmap_rel" | "mcpr">(
    "heatmap_abs"
  );
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // MCPR 用パラメータ
  const [blankIndex, setBlankIndex] = useState("2");
  const [timespanSec, setTimespanSec] = useState(180);
  const [lowerOD, setLowerOD] = useState(0.1);
  const [upperOD, setUpperOD] = useState(0.3);

  // ブレイクポイント
  const theme = useTheme();
  const isSmall = useMediaQuery(theme.breakpoints.down("sm"));

  /* -----------------------------
   * Event handlers
   * --------------------------- */
  const handleModeChange = (event: SelectChangeEvent) => {
    setMode(event.target.value as typeof mode);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files?.length) {
      setFile(event.target.files[0]);
    }
  };

  const handleGenerateGraph = async () => {
    if (!file) {
      window.alert("Please select a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    try {
      const requestUrl =
        mode === "mcpr"
          ? `${url_prefix}/graph_engine/mcpr?blank_index=${blankIndex}&timespan_sec=${timespanSec}&lower_OD=${lowerOD}&upper_OD=${upperOD}`
          : `${url_prefix}/graph_engine/${mode}`;

      const res = await fetch(requestUrl, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Request failed");

      const blob = await res.blob();
      setImageSrc(URL.createObjectURL(blob));
    } catch (err) {
      console.error(err);
      window.alert("Failed to generate graph.");
    } finally {
      setIsLoading(false);
    }
  };

  /* -----------------------------
   * Render
   * --------------------------- */
  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      {/* パンくずリスト */}
      <Breadcrumbs sx={{ mb: 3 }}>
        <Link underline="hover" color="inherit" href="/">
          Top
        </Link>
        <Typography color="text.primary">Graph engine</Typography>
      </Breadcrumbs>

      <Paper
        variant="outlined"
        sx={{
          p: { xs: 2, sm: 4 },
          borderRadius: 3,
          backgroundColor: theme.palette.mode === "light" ? "#fafafa" : "#121212",
        }}
      >
        <Stack spacing={3}>
          {/* グラフモード選択 */}
          <FormControl fullWidth>
            <InputLabel id="mode-select-label">Graph Mode</InputLabel>
            <Select
              labelId="mode-select-label"
              value={mode}
              label="Graph Mode"
              onChange={handleModeChange}
              disabled={isLoading}
              size="small"
            >
              <MenuItem value="heatmap_abs">Heatmap (abs.)</MenuItem>
              <MenuItem value="heatmap_rel">Heatmap (rel.)</MenuItem>
              <MenuItem value="mcpr">MCPR</MenuItem>
            </Select>
          </FormControl>

          {/* MCPR parameters */}
          {mode === "mcpr" && (
            <Stack
              direction={isSmall ? "column" : "row"}
              spacing={2}
              divider={<Divider orientation="vertical" flexItem />}
            >
              {/* 非表示の blankIndex。必要であれば UI に戻してください */}
              {/* <TextField
                label="blank_index"
                type="number"
                value={blankIndex}
                onChange={(e) => setBlankIndex(e.target.value)}
                disabled={isLoading}
                fullWidth
                size="small"
              /> */}

              <TextField
                label="interval (s)"
                type="number"
                value={timespanSec}
                onChange={(e) => setTimespanSec(Number(e.target.value))}
                disabled={isLoading}
                fullWidth
                size="small"
              />
              <TextField
                label="lower OD"
                type="number"
                inputProps={{ step: "0.01" }}
                value={lowerOD}
                onChange={(e) => setLowerOD(Number(e.target.value))}
                disabled={isLoading}
                fullWidth
                size="small"
              />
              <TextField
                label="upper OD"
                type="number"
                inputProps={{ step: "0.01" }}
                value={upperOD}
                onChange={(e) => setUpperOD(Number(e.target.value))}
                disabled={isLoading}
                fullWidth
                size="small"
              />
            </Stack>
          )}

          {/* ファイル選択 & 実行ボタン */}
          <Stack
            direction={isSmall ? "column" : "row"}
            spacing={2}
            alignItems="stretch"
          >
            <Box sx={{ flex: 1 }}>
              <label style={{ display: "block" }}>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                />
                <BlackButton as="span" disabled={isLoading}>
                  {file ? file.name : "Select CSV"}
                </BlackButton>
              </label>
            </Box>

            <Box sx={{ flex: 1 }}>
              <BlackButton
                type="button"
                disabled={isLoading}
                onClick={handleGenerateGraph}
              >
                {isLoading && (
                  <CircularProgress
                    size={20}
                    sx={{ color: "#fff" }}
                    thickness={4}
                  />
                )}
                {isLoading ? "Generating..." : "Generate Graph"}
              </BlackButton>
            </Box>
          </Stack>

          {/* 生成結果 */}
          {imageSrc && (
            <Box>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Generated Graph
              </Typography>
              <Box
                component="img"
                src={imageSrc}
                alt="Generated Graph"
                sx={{
                  width: "100%",
                  maxHeight: "70vh",
                  objectFit: "contain",
                  borderRadius: 2,
                  boxShadow: 3,
                }}
              />
            </Box>
          )}
        </Stack>
      </Paper>
    </Container>
  );
};

export default GraphEngine;
