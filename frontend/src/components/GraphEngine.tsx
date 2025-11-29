// src/components/GraphEngine.tsx
import React, { useCallback, useEffect, useState } from "react";
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
  Autocomplete,
} from "@mui/material";
import { styled, useTheme } from "@mui/material/styles";
import axios from "axios";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

/* -------------------------------------------------
 * UI helpers
 * ------------------------------------------------- */

const StyledButton = styled("button")(({ theme }) => ({
  appearance: "none",
  border: "none",
  outline: "none",
  padding: 0,
  margin: 0,
  font: "inherit",
  width: "100%",
  minHeight: 40,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: theme.spacing(1),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
  fontWeight: 500,
  letterSpacing: 0.5,
  cursor: "pointer",
  transition: "background-color 0.2s ease",
  "&:hover": { backgroundColor: theme.palette.primary.dark },
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
  const [mode, setMode] = useState<
    "heatmap_abs" | "heatmap_rel" | "mcpr" | "cell_lengths"
  >("heatmap_abs");
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dbNames, setDbNames] = useState<string[]>([]);
  const [selectedDb, setSelectedDb] = useState("");
  const [dbInputValue, setDbInputValue] = useState("");
  const [dbLoading, setDbLoading] = useState(false);
  const [dbError, setDbError] = useState<string | null>(null);
  const [selectedLabel, setSelectedLabel] = useState("1");

  // MCPR 用パラメータ（★ 数値入力は文字列で保持）
  const [blankIndex, setBlankIndex] = useState("2");
  const [timespanSec, setTimespanSec] = useState("180"); // ★
  const [lowerOD, setLowerOD] = useState("0.1");         // ★
  const [upperOD, setUpperOD] = useState("0.3");         // ★

  const theme = useTheme();
  const isSmall = useMediaQuery(theme.breakpoints.down("sm"));

  /* -----------------------------
   * Event handlers
   * --------------------------- */
  const handleModeChange = (event: SelectChangeEvent) => {
    setMode(event.target.value as typeof mode);
  };

  const fetchDbNames = useCallback(async () => {
    setDbLoading(true);
    setDbError(null);
    try {
      const token = localStorage.getItem("access_token");
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const res = await axios.get(`${url_prefix}/databases`, {
        headers,
        params: { page_size: 200, display_mode: "All" },
      });
      const names = Array.isArray(res.data?.databases)
        ? res.data.databases
        : Array.isArray(res.data)
        ? res.data
        : [];
      setDbNames(names);
      if (!selectedDb && names.length > 0) {
        setSelectedDb(names[0]);
        setDbInputValue(names[0]);
      } else if (selectedDb && !names.includes(selectedDb) && names.length > 0) {
        setSelectedDb(names[0]);
        setDbInputValue(names[0]);
      }
    } catch (err) {
      console.error(err);
      setDbError("Failed to load databases.");
    } finally {
      setDbLoading(false);
    }
  }, [selectedDb]);

  useEffect(() => {
    if (mode === "cell_lengths" && dbNames.length === 0 && !dbLoading) {
      fetchDbNames();
    }
  }, [dbLoading, dbNames.length, fetchDbNames, mode]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files?.length) setFile(event.target.files[0]);
  };

  // ★ 共通数値入力ハンドラ（空文字も許容）
  const handleNumberChange =
    (setter: React.Dispatch<React.SetStateAction<string>>) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = e.target.value;
      // 先頭 0 は自動除去したい場合は下行を有効に
      // const normalized = v.replace(/^0+(?=\d)/, "");
      setter(v);
    };

  const handleGenerateGraph = async () => {
    if (mode === "cell_lengths") {
      if (!selectedDb) {
        window.alert("Please select a database first.");
        return;
      }
      setIsLoading(true);
      setImageSrc(null);
      try {
        const requestUrl = `${url_prefix}/graph_engine/cell_lengths?db_name=${encodeURIComponent(
          selectedDb
        )}&label=${encodeURIComponent(selectedLabel)}`;
        const res = await fetch(requestUrl);
        if (res.status === 404) {
          window.alert("No cells found for the selected label.");
          return;
        }
        if (!res.ok) throw new Error("Request failed");
        const blob = await res.blob();
        setImageSrc(URL.createObjectURL(blob));
      } catch (err) {
        console.error(err);
        window.alert("Failed to generate cell length plot.");
      } finally {
        setIsLoading(false);
      }
      return;
    }

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
          ? `${url_prefix}/graph_engine/mcpr?blank_index=${blankIndex}&timespan_sec=${Number(
              timespanSec || "0"
            )}&lower_OD=${Number(lowerOD || "0")}&upper_OD=${Number(
              upperOD || "0"
            )}`
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
              <MenuItem value="cell_lengths">Cell lengths</MenuItem>
            </Select>
          </FormControl>

          {/* MCPR parameters */}
          {mode === "mcpr" && (
            <Stack
              direction={isSmall ? "column" : "row"}
              spacing={2}
              divider={<Divider orientation="vertical" flexItem />}
            >
              <TextField
                label="interval (s)"
                type="number"
                value={timespanSec}               // ★ string
                onChange={handleNumberChange(setTimespanSec)} // ★
                disabled={isLoading}
                fullWidth
                size="small"
              />
              <TextField
                label="lower OD"
                type="number"
                inputProps={{ step: "0.01" }}
                value={lowerOD}                   // ★
                onChange={handleNumberChange(setLowerOD)}     // ★
                disabled={isLoading}
                fullWidth
                size="small"
              />
              <TextField
                label="upper OD"
                type="number"
                inputProps={{ step: "0.01" }}
                value={upperOD}                   // ★
                onChange={handleNumberChange(setUpperOD)}     // ★
                disabled={isLoading}
                fullWidth
                size="small"
              />
            </Stack>
          )}

          {/* DB & label selection for cell length mode */}
          {mode === "cell_lengths" && (
            <Stack
              direction={isSmall ? "column" : "row"}
              spacing={2}
              alignItems="stretch"
            >
              <Box sx={{ flex: 2, minWidth: 320 }}>
                <Autocomplete
                  fullWidth
                  freeSolo
                  options={dbNames}
                  value={selectedDb}
                  inputValue={dbInputValue}
                  onChange={(_, newValue) => {
                    setSelectedDb(newValue || "");
                    setDbInputValue(newValue || "");
                  }}
                  onInputChange={(_, newInput) => {
                    setSelectedDb(newInput);
                    setDbInputValue(newInput);
                  }}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      label="Database"
                      size="small"
                      fullWidth
                      disabled={dbLoading || isLoading}
                      placeholder="Type or pick a database name"
                    />
                  )}
                />
              </Box>

              <FormControl
                size="small"
                sx={{ minWidth: 140, flex: 1 }}
                disabled={isLoading}
              >
                <InputLabel id="label-select-label">Label</InputLabel>
                <Select
                  labelId="label-select-label"
                  value={selectedLabel}
                  label="Label"
                  onChange={(e) => setSelectedLabel(e.target.value)}
                >
                  <MenuItem value="N/A">N/A</MenuItem>
                  {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
                    <MenuItem key={n} value={String(n)}>
                      {n}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Stack>
          )}
          {dbError && mode === "cell_lengths" && (
            <Typography color="error" variant="body2">
              {dbError} You can still type a database name directly.
            </Typography>
          )}

          {/* ファイル選択 & 実行ボタン */}
          <Stack
            direction={isSmall ? "column" : "row"}
            spacing={2}
            alignItems="stretch"
          >
            {mode !== "cell_lengths" && (
              <Box sx={{ flex: 1 }}>
                <label style={{ display: "block" }}>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    style={{ display: "none" }}
                  />
                  <StyledButton as="span" disabled={isLoading}>
                    {file ? file.name : "Select CSV"}
                  </StyledButton>
                </label>
              </Box>
            )}

            <Box sx={{ flex: 1 }}>
              <StyledButton
                type="button"
                disabled={
                  isLoading ||
                  (mode === "cell_lengths" && (dbLoading || !selectedDb))
                }
                onClick={handleGenerateGraph}
              >
                {isLoading && (
                  <CircularProgress
                    size={20}
                    sx={{ color: theme.palette.primary.contrastText }}
                    thickness={4}
                  />
                )}
                {isLoading ? "Generating..." : "Generate Graph"}
              </StyledButton>
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
