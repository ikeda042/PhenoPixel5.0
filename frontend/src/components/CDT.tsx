import React, { useState } from "react";
import {
  Box,
  Container,
  Breadcrumbs,
  Link,
  Typography,
  Button,
  CircularProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import axios from "axios";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

interface ResultItem {
  filename: string;
  mean_length: number;
  nagg_rate: number;
}

const BlackButton = styled("button")(({ theme }) => ({
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
  backgroundColor: "#000",
  color: "#fff",
  fontWeight: 500,
  letterSpacing: 0.5,
  cursor: "pointer",
  transition: "background-color 0.2s ease",
  "&:hover": { backgroundColor: "#222" },
  "&:disabled": {
    backgroundColor: theme.palette.action.disabledBackground,
    color: theme.palette.action.disabled,
    cursor: "not-allowed",
  },
}));

const CDT: React.FC = () => {
  const [ctrlFile, setCtrlFile] = useState<File | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [heatmaps, setHeatmaps] = useState<
    Record<string, { abs: string; rel: string; dist: string }>
  >({});

  const handleCtrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) setCtrlFile(e.target.files[0]);
  };

  const handleFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) setFiles(e.target.files);
  };

  const handleAnalyze = async () => {
    if (!ctrlFile || !files?.length) {
      alert("Please select files");
      return;
    }
    setIsLoading(true);
    const formData = new FormData();
    formData.append("ctrl_file", ctrlFile);
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    try {
      const res = await axios.post<ResultItem[]>(`${url_prefix}/cdt/nagg`, formData);
      setResults(res.data);

      // Generate heatmaps and distribution for each file
      const heatmapPromises = Array.from(files).map(async (file) => {
        const fdAbs = new FormData();
        fdAbs.append("file", file);
        const absRes = await fetch(`${url_prefix}/graph_engine/heatmap_abs`, {
          method: "POST",
          body: fdAbs,
        });
        if (!absRes.ok) throw new Error("Failed to generate abs heatmap");
        const absBlob = await absRes.blob();

        const fdRel = new FormData();
        fdRel.append("file", file);
        const relRes = await fetch(`${url_prefix}/graph_engine/heatmap_rel`, {
          method: "POST",
          body: fdRel,
        });
        if (!relRes.ok) throw new Error("Failed to generate rel heatmap");
        const relBlob = await relRes.blob();

        const fdDist = new FormData();
        fdDist.append("file", file);
        const distRes = await fetch(`${url_prefix}/graph_engine/distribution`, {
          method: "POST",
          body: fdDist,
        });
        if (!distRes.ok) throw new Error("Failed to generate distribution");
        const distBlob = await distRes.blob();

        return [
          file.name,
          {
            abs: URL.createObjectURL(absBlob),
            rel: URL.createObjectURL(relBlob),
            dist: URL.createObjectURL(distBlob),
          },
        ] as [string, { abs: string; rel: string; dist: string }];
      });

      const heatmapEntries = await Promise.allSettled(heatmapPromises);
      const map: Record<string, { abs: string; rel: string; dist: string }> = {};
      for (const h of heatmapEntries) {
        if (h.status === "fulfilled") {
          const [name, urls] = h.value;
          map[name] = urls;
        }
      }
      setHeatmaps(map);
    } catch (err) {
      console.error(err);
      alert("Failed to analyze files.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container sx={{ py: 4 }}>
      <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 3 }}>
        <Link underline="hover" color="inherit" href="/">
          Top
        </Link>
        <Typography color="text.primary">CDT</Typography>
      </Breadcrumbs>

      <Paper
        variant="outlined"
        sx={{
          p: { xs: 2, sm: 4 },
          borderRadius: 3,
          backgroundColor: "#fafafa",
        }}
      >
        <Stack spacing={3}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
            <Box sx={{ flex: 1 }}>
              <label style={{ display: "block" }}>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleCtrlChange}
                  style={{ display: "none" }}
                />
                <BlackButton as="span" disabled={isLoading}>
                  {ctrlFile ? ctrlFile.name : "Select control CSV"}
                </BlackButton>
              </label>
            </Box>
            <Box sx={{ flex: 1 }}>
              <label style={{ display: "block" }}>
                <input
                  type="file"
                  accept=".csv"
                  multiple
                  onChange={handleFilesChange}
                  style={{ display: "none" }}
                />
                <BlackButton as="span" disabled={isLoading}>
                  {files ? `${files.length} file(s) selected` : "Select CSV files"}
                </BlackButton>
              </label>
            </Box>
          </Stack>

          <Box>
            <BlackButton onClick={handleAnalyze} disabled={isLoading}>
              {isLoading && (
                <CircularProgress size={20} sx={{ color: "#fff" }} thickness={4} />
              )}
              {isLoading ? "Analyzing..." : "Analyze"}
            </BlackButton>
          </Box>

          {results.length > 0 && (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Filename</TableCell>
                    <TableCell>Mean Length (μm)</TableCell>
                    <TableCell>Nagg Rate</TableCell>
                    <TableCell>Heatmap (abs.)</TableCell>
                    <TableCell>Heatmap (rel.)</TableCell>
                    <TableCell>Distribution</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {results.map((r, idx) => (
                    <TableRow key={idx}>
                      <TableCell>{r.filename}</TableCell>
                      <TableCell>{r.mean_length.toFixed(2)}</TableCell>
                      <TableCell>{(r.nagg_rate * 100).toFixed(2)}%</TableCell>
                      <TableCell>
                        {heatmaps[r.filename]?.abs && (
                          <Box
                            component="img"
                            src={heatmaps[r.filename].abs}
                            alt="heatmap abs"
                            sx={{ width: 120, borderRadius: 1 }}
                          />
                        )}
                      </TableCell>
                      <TableCell>
                        {heatmaps[r.filename]?.rel && (
                          <Box
                            component="img"
                            src={heatmaps[r.filename].rel}
                            alt="heatmap rel"
                            sx={{ width: 120, borderRadius: 1 }}
                          />
                        )}
                      </TableCell>
                      <TableCell>
                        {heatmaps[r.filename]?.dist && (
                          <Box
                            component="img"
                            src={heatmaps[r.filename].dist}
                            alt="distribution"
                            sx={{ width: 120, borderRadius: 1 }}
                          />
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Stack>
      </Paper>
    </Container>
  );
};

export default CDT;
