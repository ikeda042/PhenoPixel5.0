import React, { useEffect, useState } from "react";
import { Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, IconButton, TextField, Button, Grid, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Select, MenuItem, SelectChangeEvent, Link, Breadcrumbs } from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import DatabaseIcon from '@mui/icons-material/Storage';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import { useNavigate } from "react-router-dom";
import { useLocation } from "react-router-dom";
import TaskIcon from '@mui/icons-material/Task';

interface ListDBResponse {
    databases: string[];
}

const url_prefix = settings.url_prefix;

const Databases: React.FC = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const default_search_word = queryParams.get('default_search_word') ?? "";
    const [databases, setDatabases] = useState<string[]>([]);
    const [searchQuery, setSearchQuery] = useState(default_search_word);
    const [displayMode, setDisplayMode] = useState('User uploaded');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [dialogMessage, setDialogMessage] = useState("");
    const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
    const [databaseToComplete, setDatabaseToComplete] = useState<string | null>(null);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchDatabases = async () => {
            try {
                const response = await axios.get<ListDBResponse>(`${url_prefix}/databases`);
                setDatabases(response.data.databases);
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
        setDisplayMode(event.target.value);
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

    const filteredDatabases = databases.filter(database => {
        const searchMatch = database.toLowerCase().includes(searchQuery.toLowerCase());
        if (displayMode === 'User uploaded') {
            return searchMatch && database.endsWith('-uploaded.db');
        }
        if (displayMode === 'Completed') {
            return searchMatch && database.endsWith('-completed.db');
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
                    <Typography color="text.primary">Database Console</Typography>
                </Breadcrumbs>
            </Box>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="space-between" mt={2}>
                <Grid container spacing={2} alignItems="center">
                    <Grid item xs={4}>
                        <TextField
                            label="Search Database"
                            variant="outlined"
                            fullWidth
                            value={searchQuery}
                            onChange={handleSearchChange}
                            sx={{ height: '56px' }}
                        />
                    </Grid>
                    <Grid item xs={2}>
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
                    <Grid item xs={3}>
                        <input
                            accept=".db"
                            style={{ display: 'none' }}
                            id="raised-button-file"
                            multiple
                            type="file"
                            onChange={handleFileChange}
                        />
                        <label htmlFor="raised-button-file">
                            <Button
                                variant="contained"
                                component="span"
                                startIcon={<DatabaseIcon />}
                                sx={{
                                    backgroundColor: 'white',
                                    color: 'black',
                                    width: '100%',
                                    height: '56px',
                                    '&:hover': {
                                        backgroundColor: 'lightgrey'
                                    }
                                }}
                            >
                                {selectedFile ? selectedFile.name : "Select Database"}
                            </Button>
                        </label>
                    </Grid>
                    <Grid item xs={3}>
                        <Button
                            onClick={handleUpload}
                            variant="contained"
                            sx={{
                                backgroundColor: 'black',
                                color: 'white',
                                width: '100%',
                                height: '56px',
                                '&:hover': {
                                    backgroundColor: 'grey'
                                }
                            }}
                            startIcon={<FileUploadIcon />}
                            disabled={!selectedFile}
                        >
                            Upload
                        </Button>
                    </Grid>
                </Grid>
            </Box>

            <Box mt={3}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Database Name</TableCell>
                                {displayMode === 'User uploaded' && <TableCell align="center">Mark as Complete</TableCell>}
                                <TableCell align="center">Go</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredDatabases.map((database, index) => (
                                <TableRow key={index}>
                                    <TableCell component="th" scope="row">
                                        {database}
                                    </TableCell>
                                    {displayMode === 'User uploaded' && (
                                        <TableCell align="center">
                                            <Button
                                                variant="contained"
                                                sx={{
                                                    backgroundColor: 'green',
                                                    color: 'white',
                                                    '&:hover': {
                                                        backgroundColor: 'darkgreen'
                                                    }
                                                }}
                                                onClick={() => handleOpenConfirmDialog(database)}
                                                startIcon={<TaskIcon />}
                                            >
                                                Mark as Complete
                                            </Button>
                                        </TableCell>
                                    )}
                                    <TableCell align="right">
                                        <IconButton onClick={() => handleNavigate(database)}>
                                            <Typography>Access database </Typography>
                                            <NavigateNextIcon />
                                        </IconButton>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Box>

            <Dialog open={dialogOpen} onClose={handleCloseDialog}>
                <DialogTitle>{"File Upload Status"}</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        {dialogMessage}
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseDialog} color="primary">
                        Close
                    </Button>
                </DialogActions>
            </Dialog>

            <Dialog open={confirmDialogOpen} onClose={handleCloseConfirmDialog}>
                <DialogTitle>{"Confirm Mark as Complete"}</DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Are you sure you want to mark this database as complete?
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseConfirmDialog} color="primary">
                        Cancel
                    </Button>
                    <Button onClick={handleMarkAsComplete} color="primary">
                        Confirm
                    </Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
};

export default Databases;
