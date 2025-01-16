import React, { useState } from "react";
import {
  Box,
  Button,
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
  TextField
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const GraphEngine: React.FC = () => {
  const [mode, setMode] = useState("heatmap_abs");
  const [file, setFile] = useState<File | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // MCPR用パラメータ
  const [blankIndex, setBlankIndex] = useState("2");
  const [timespanSec, setTimespanSec] = useState(180);
  const [lowerOD, setLowerOD] = useState(0.1);
  const [upperOD, setUpperOD] = useState(0.3);

  // ブレイクポイントに応じたレスポンシブ制御
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  const handleModeChange = (event: SelectChangeEvent<string>) => {
    setMode(event.target.value as string);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleGenerateGraph = async () => {
    if (!file) {
      alert("Please select a CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);

    try {
      let requestUrl;
      if (mode === "mcpr") {
        // MCPRモード時はURLにクエリパラメータを付与
        requestUrl = `${url_prefix}/graph_engine/mcpr?blank_index=${blankIndex}&timespan_sec=${timespanSec}&lower_OD=${lowerOD}&upper_OD=${upperOD}`;
      } else {
        requestUrl = `${url_prefix}/graph_engine/${mode}`;
      }

      const response = await fetch(requestUrl, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setImageSrc(imageUrl);
      } else {
        alert("Failed to generate graph.");
      }
    } catch (error) {
      console.error("Error generating graph:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 2 }}>
      {/* パンくずリスト */}
      <Box mb={2}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Typography color="text.primary">Graph engine</Typography>
        </Breadcrumbs>
      </Box>

      <Paper elevation={3} sx={{ p: 3 }}>
        {/* グラフモード選択 */}
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel id="select-label">Graph Mode</InputLabel>
          <Select
            labelId="select-label"
            value={mode}
            onChange={handleModeChange}
            label="Graph Mode"
            disabled={isLoading}
          >
            <MenuItem value="heatmap_abs">Heatmap abs.</MenuItem>
            <MenuItem value="heatmap_rel">Heatmap rel.</MenuItem>
            <MenuItem value="mcpr">MCPR</MenuItem>
          </Select>
        </FormControl>

        {/* MCPRモードの場合に追加パラメータを表示 */}
        {mode === "mcpr" && (
          <Box
            sx={{
              display: "flex",
              flexDirection: isSmallScreen ? "column" : "row",
              gap: 2,
              mb: 2,
            }}
          >
            <FormControl sx={{ flex: 1 }}>
              <TextField
                label="blank_index"
                type="number"
                value={blankIndex}
                onChange={(e) => setBlankIndex(e.target.value)}
                disabled={isLoading}
              />
            </FormControl>

            <FormControl sx={{ flex: 1 }}>
              <TextField
                label="timespan_sec"
                type="number"
                value={timespanSec}
                onChange={(e) => setTimespanSec(Number(e.target.value))}
                disabled={isLoading}
              />
            </FormControl>

            <FormControl sx={{ flex: 1 }}>
              <TextField
                label="lower_OD"
                type="number"
                inputProps={{ step: "0.01" }}
                value={lowerOD}
                onChange={(e) => setLowerOD(Number(e.target.value))}
                disabled={isLoading}
              />
            </FormControl>

            <FormControl sx={{ flex: 1 }}>
              <TextField
                label="upper_OD"
                type="number"
                inputProps={{ step: "0.01" }}
                value={upperOD}
                onChange={(e) => setUpperOD(Number(e.target.value))}
                disabled={isLoading}
              />
            </FormControl>
          </Box>
        )}

        {/* ファイル選択＋グラフ生成ボタン */}
        <Box
          sx={{
            display: "flex",
            flexDirection: isSmallScreen ? "column" : "row",
            alignItems: "center",
            gap: 2,
            mb: 2,
          }}
        >
          <Button variant="outlined" component="label" sx={{ width: isSmallScreen ? "100%" : "auto" }}>
            Select CSV File
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              hidden
            />
          </Button>

          <Button
            variant="contained"
            color="primary"
            onClick={handleGenerateGraph}
            disabled={isLoading}
            sx={{
              width: isSmallScreen ? "100%" : "auto",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {isLoading ? (
              <>
                <CircularProgress size={24} sx={{ color: "#fff", mr: 1 }} />
                Generating...
              </>
            ) : (
              "Generate Graph"
            )}
          </Button>
        </Box>

        {/* 生成結果の画像表示 */}
        {imageSrc && (
          <Box mt={4}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Generated Graph:
            </Typography>
            <Box
              component="img"
              src={imageSrc}
              alt="Generated Graph"
              sx={{
                width: "100%",
                maxHeight: "80vh",
                display: "block",
                objectFit: "contain",
                borderRadius: 1,
              }}
            />
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default GraphEngine;
