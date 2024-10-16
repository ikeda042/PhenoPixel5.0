import React, { useEffect, useState } from "react";
import { Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, TextField, Select, MenuItem, Link } from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

interface FileItem {
    name: string;
}

const url_prefix = settings.url_prefix;

const ResultsConsole: React.FC = () => {
    const [files, setFiles] = useState<FileItem[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [displayMode, setDisplayMode] = useState(() => localStorage.getItem('displayMode') || 'User uploaded');

    useEffect(() => {
        const fetchFiles = async () => {
            try {
                const response = await axios.get<FileItem[]>(`${url_prefix}/results`);
                setFiles(response.data); // API response is already an array of {name: string}
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

    const filteredFiles = files.filter(file =>
        file.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <Container>
            <Box>
                <Typography variant="h4">Results Console</Typography>
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
                    value={displayMode}
                    onChange={(e) => setDisplayMode(e.target.value)}
                    displayEmpty
                    inputProps={{ 'aria-label': 'Without label' }}
                    sx={{ width: '28%' }}
                >
                    <MenuItem value="Validated">Validated</MenuItem>
                    <MenuItem value="User uploaded">Uploaded</MenuItem>
                    <MenuItem value="Completed">Completed</MenuItem>
                </Select>
            </Box>

            <Box mt={3}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>File Name</TableCell>
                                <TableCell align="center">Download</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredFiles.map((file, index) => (
                                <TableRow key={index}>
                                    <TableCell>
                                        <Link component="button" variant="body2" onClick={() => handleDownload(file.name)}>
                                            {file.name}
                                        </Link>
                                    </TableCell>
                                    <TableCell align="center">
                                        <Link component="button" onClick={() => handleDownload(file.name)}>
                                            Download
                                        </Link>
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
