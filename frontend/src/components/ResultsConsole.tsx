import React, { useEffect, useState } from "react";
import { Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, IconButton, Button, Grid, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Select, MenuItem, SelectChangeEvent, Link, Breadcrumbs, CircularProgress, TextField, Tooltip } from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import DatabaseIcon from '@mui/icons-material/Storage';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import { useNavigate } from "react-router-dom";
import { useLocation } from "react-router-dom";
import TaskIcon from '@mui/icons-material/Task';
import DownloadIcon from '@mui/icons-material/Download';

interface ListDBResponse {
    databases: string[];
}

const url_prefix = settings.url_prefix;

const ResultsConsole: React.FC = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const default_search_word = queryParams.get('default_search_word') ?? "";
    const [databases, setDatabases] = useState<string[]>([]);
    const [searchQuery, setSearchQuery] = useState(default_search_word);
    const [displayMode, setDisplayMode] = useState(() => localStorage.getItem('displayMode') || 'User uploaded');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [dialogMessage, setDialogMessage] = useState("");
    const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
    const [databaseToComplete, setDatabaseToComplete] = useState<string | null>(null);
    const [markableDatabases, setMarkableDatabases] = useState<{ [key: string]: boolean }>({});
    const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
    const [previewImage, setPreviewImage] = useState<string | null>(null);
    const [selectedMode, setSelectedMode] = useState("fluo");
    const [selectedLabel, setSelectedLabel] = useState("1");
    const [loadingPreview, setLoadingPreview] = useState(false);
    const [metadata, setMetadata] = useState<{ [key: string]: string }>({});
    const [newMetadata, setNewMetadata] = useState<{ [key: string]: string }>({});
    const navigate = useNavigate();

    useEffect(() => {
        const fetchDatabases = async () => {
            try {
                const response = await axios.get<ListDBResponse>(`${url_prefix}/databases`);
                setDatabases(response.data.databases);

                const uploadedDatabases = response.data.databases.filter(db => db.endsWith('-uploaded.db'));

                const markableStatus = await Promise.all(
                    uploadedDatabases.map(async (db) => {
                        const checkResponse = await axios.get(`${url_prefix}/databases/${db}`);
                        return { db, markable: checkResponse.data };
                    })
                );

                const markableStatusMap = markableStatus.reduce((acc, { db, markable }) => {
                    acc[db] = markable;
                    return acc;
                }, {} as { [key: string]: boolean });

                setMarkableDatabases(markableStatusMap);

                // Fetch metadata for each database
                const metadataResponses = await Promise.all(
                    response.data.databases.map(async (db) => {
                        const metadataResponse = await axios.get(`${url_prefix}/databases/${db}/metadata`);
                        return { db, metadata: metadataResponse.data };
                    })
                );

                const metadataMap = metadataResponses.reduce((acc, { db, metadata }) => {
                    acc[db] = metadata;
                    return acc;
                }, {} as { [key: string]: string });

                setMetadata(metadataMap);
                setNewMetadata(metadataMap);
            } catch (error) {
                console.error("Failed to fetch databases", error);
            }
        };

        fetchDatabases();
    }, []);

    const handleNavigate = (dbName: string) => {
        navigate(`/databases/?db_name=${dbName}`);
    };

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchQuery(event.target.value);
    };

    const handleDisplayModeChange = (event: SelectChangeEvent<string>) => {
        const newDisplayMode = event.target.value;
        setDisplayMode(newDisplayMode);
        localStorage.setItem('displayMode', newDisplayMode);
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            setSelectedFile(event.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (selectedFile) {
            const formData = new FormData();
            formData.append("file", selectedFile);
            try {
                const response = await axios.post(`${url_prefix}/databases/upload`, formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });
                setDialogMessage("Database uploaded successfully!");
                setDialogOpen(true);
                console.log(response.data);
            } catch (error) {
                setDialogMessage("Failed to upload database.");
                setDialogOpen(true);
                console.error("Failed to upload database", error);
            }
        }
    };

    const handleCloseDialog = () => {
        setDialogOpen(false);
        window.location.reload();
    };

    const handleOpenConfirmDialog = (database: string) => {
        setDatabaseToComplete(database);
        setConfirmDialogOpen(true);
    };

    const handleCloseConfirmDialog = () => {
        setConfirmDialogOpen(false);
        setDatabaseToComplete(null);
    };

    const handleMarkAsComplete = async () => {
        if (databaseToComplete) {
            try {
                const response = await axios.patch(`${url_prefix}/databases/${databaseToComplete}`);
                setDialogMessage(response.data.message);
                setDialogOpen(true);
                handleCloseConfirmDialog();
            } catch (error) {
                setDialogMessage("Failed to mark database as complete.");
                setDialogOpen(true);
                console.error("Failed to mark database as complete", error);
            }
        }
    };

    const handleDownload = async (database: string) => {
        try {
            const response = await axios.get(`${url_prefix}/databases/download-completed/${database}`, {
                responseType: 'blob'
            });

            const contentDisposition = response.headers['content-disposition'];
            const fileName = contentDisposition ? contentDisposition.split('filename=')[1] : database;

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', fileName);
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            setDialogMessage("Failed to download database.");
            setDialogOpen(true);
            console.error("Failed to download database", error);
        }
    };

    const handlePreview = async (database: string) => {
        setLoadingPreview(true);
        try {
            const response = await axios.get(`${url_prefix}/databases/${database}/combined_images`, {
                params: {
                    label: selectedLabel,
                    mode: selectedMode
                },
                responseType: 'blob'
            });

            const url = window.URL.createObjectURL(new Blob([response.data]));
            setPreviewImage(url);
            setPreviewDialogOpen(true);
        } catch (error) {
            setDialogMessage("Failed to fetch preview image.");
            setDialogOpen(true);
            console.error("Failed to fetch preview image", error);
        } finally {
            setLoadingPreview(false);
        }
    };

    const handleClosePreviewDialog = () => {
        setPreviewDialogOpen(false);
        setPreviewImage(null);
    };

    const handleMetadataChange = async (dbName: string, newMetadata: string) => {
        setNewMetadata(prevMetadata => ({
            ...prevMetadata,
            [dbName]: newMetadata
        }));

        try {
            await axios.patch(`${url_prefix}/databases/${dbName}/update-metadata`, {
                metadata: newMetadata
            }, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            setDialogMessage("Metadata updated successfully!");
            setDialogOpen(true);
            setMetadata(prevMetadata => ({
                ...prevMetadata,
                [dbName]: newMetadata
            }));
        } catch (error) {
            setDialogMessage("Failed to update metadata.");
            setDialogOpen(true);
            console.error("Failed to update metadata", error);
        }
    };

    const filteredDatabases = databases.filter(database => {
        const searchMatch = database.toLowerCase().includes(searchQuery.toLowerCase());
        if (displayMode === 'User uploaded') {
            return searchMatch && database.endsWith('-uploaded.db');
        }
        if (displayMode === 'Completed') {
            return searchMatch && database.endsWith('-completed.db');
        }
        if (displayMode === 'Validated') {
            return searchMatch && !database.endsWith('-uploaded.db') && !database.endsWith('-completed.db');
        }
        return searchMatch;
    });

    return (
        <Container>
            <Box>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Typography color="text.primary">Results</Typography>
                </Breadcrumbs>
            </Box>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="space-between" mt={2}>
                <Grid container spacing={2} alignItems="center">
                    <Grid item xs={8}>
                        <TextField
                            label="Search Files"
                            variant="outlined"
                            fullWidth
                            value={searchQuery}
                            onChange={handleSearchChange}
                            sx={{ height: '56px' }}
                        />
                    </Grid>
                    <Grid item xs={4}>
                        <Select
                            value={displayMode}
                            onChange={handleDisplayModeChange}
                            displayEmpty
                            inputProps={{ 'aria-label': 'Without label' }}
                            fullWidth
                            sx={{ height: '56px' }}
                        >
                            <MenuItem value="Validated">Validated</MenuItem>
                            <MenuItem value="User uploaded">Uploaded</MenuItem>
                            <MenuItem value="Completed">Completed</MenuItem>
                        </Select>
                    </Grid>

                </Grid>
            </Box>

            <Box mt={3}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Files</TableCell>
                                <TableCell align="center"></TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredDatabases.map((database, index) => (
                                <TableRow key={index}>

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
