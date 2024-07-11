import React, { useEffect, useState } from "react";
import { Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, IconButton, TextField, Button, Grid, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import DatabaseIcon from '@mui/icons-material/Storage';
import { useNavigate } from "react-router-dom";
import FileUploadIcon from '@mui/icons-material/FileUpload';

interface ListDBResponse {
    databases: string[];
}

const url_prefix = settings.url_prefix;

const TopPage: React.FC = () => {
    const [databases, setDatabases] = useState<string[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [dialogMessage, setDialogMessage] = useState("");
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
    };

    const filteredDatabases = databases.filter(database => database.toLowerCase().includes(searchQuery.toLowerCase()));

    return (
        <Container>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="space-between">
                <Grid container spacing={2} alignItems="center">
                    <Grid item xs={6}>
                        <TextField
                            label="Search Database"
                            variant="outlined"
                            fullWidth
                            value={searchQuery}
                            onChange={handleSearchChange}
                            sx={{ height: '56px' }}
                        />
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
                                <TableCell align="right">Go</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredDatabases.map((database, index) => (
                                <TableRow key={index}>
                                    <TableCell component="th" scope="row">
                                        {database}
                                    </TableCell>
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
        </Container>
    );
};

export default TopPage;
