import React, { useEffect, useState } from "react";
import { Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, TextField, Select, MenuItem, Button, Breadcrumbs, Link } from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

interface FileItem {
    name: string;
}

const url_prefix = settings.url_prefix;

const ResultsConsole: React.FC = () => {
    const [files, setFiles] = useState<FileItem[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [fileExtension, setFileExtension] = useState("");

    useEffect(() => {
        const fetchFiles = async () => {
            try {
                const response = await axios.get<FileItem[]>(`${url_prefix}/results`);
                setFiles(response.data);
            } catch (error) {
                console.error("Failed to fetch files", error);
            }
        };

        fetchFiles();
    }, []);

    const handleDownload = async (fileName: string) => {
        try {
            const response = await axios.get(`${url_prefix}/results/${fileName}`, {
                responseType: 'blob'
            });

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', fileName);
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error("Failed to download file", error);
        }
    };

    const filteredFiles = files
        .filter(file =>
            file.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
            (fileExtension ? file.name.endsWith(fileExtension) : true)
        );

    return (
        <Container>
            <Box mb={3}>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Typography color="text.primary">Results console</Typography>
                </Breadcrumbs>
            </Box>

            <Box mt={3} display="flex" justifyContent="space-between">
                <TextField
                    label="Search Files"
                    variant="outlined"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    sx={{ width: '70%' }}
                />
                <Select
                    value={fileExtension}
                    onChange={(e) => setFileExtension(e.target.value as string)}
                    displayEmpty
                    inputProps={{ 'aria-label': 'Without label' }}
                    sx={{ width: '28%' }}
                >
                    <MenuItem value="">All files</MenuItem>
                    <MenuItem value=".png">.png</MenuItem>
                    <MenuItem value=".xlsx">.xlsx</MenuItem>
                    <MenuItem value=".csv">.csv</MenuItem>
                    <MenuItem value=".txt">.txt</MenuItem>
                    <MenuItem value=".gif">.gif</MenuItem>
                </Select>
            </Box>

            <Box mt={3}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell align="center">Result file</TableCell>
                                <TableCell align="center">Download</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredFiles.map((file, index) => (
                                <TableRow key={index}>
                                    <TableCell align="center">
                                        {file.name}
                                    </TableCell>
                                    <TableCell align="center">
                                        <Button variant="contained" onClick={() => handleDownload(file.name)} sx={
                                            {
                                                backgroundColor: '#4CAF50',
                                                '&:hover': {
                                                    backgroundColor: '#388e3c',
                                                }
                                            }
                                        }>
                                            Download
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Box>
        </Container>
    );
};

export default ResultsConsole;
