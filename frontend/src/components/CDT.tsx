import React, { useState } from "react";
import {
  Box,
  Container,
  Breadcrumbs,
  Link,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Stack,
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

interface ResultItem {
  filename: string;
  mean_length: number;
  nagg_rate: number;
}

const CDT: React.FC = () => {
  const [ctrlFile, setCtrlFile] = useState<File | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);

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
    } catch (err) {
      console.error(err);
      alert("Failed to analyze files.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container sx={{ py: 4 }}>
      <Breadcrumbs aria-label="breadcrumb">
        <Link underline="hover" color="inherit" href="/">
          Top
        </Link>
        <Typography color="text.primary">CDT</Typography>
      </Breadcrumbs>
      <Box mt={2} display="flex" flexDirection="column" gap={2}>
        <input type="file" accept=".csv" onChange={handleCtrlChange} />
        <input type="file" accept=".csv" multiple onChange={handleFilesChange} />
        <Button variant="contained" onClick={handleAnalyze} disabled={isLoading}>
          Analyze
        </Button>
      </Box>
      {results.length > 0 && (
        <TableContainer component={Paper} sx={{ mt: 4 }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Filename</TableCell>
                <TableCell>Mean Length</TableCell>
                <TableCell>Nagg Rate</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {results.map((r, idx) => (
                <TableRow key={idx}>
                  <TableCell>{r.filename}</TableCell>
                  <TableCell>{r.mean_length.toFixed(2)}</TableCell>
                  <TableCell>{(r.nagg_rate * 100).toFixed(2)}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Container>
  );
};

export default CDT;
