import React, { useEffect, useState } from "react";
import {
    Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper,
    IconButton, TextField, Button, Grid, Dialog, DialogActions, DialogContent, DialogContentText,
    DialogTitle, Link, Breadcrumbs, CircularProgress
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import { useNavigate } from "react-router-dom";
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';

interface ListND2FilesResponse {
    files: string[];
}

const url_prefix = settings.url_prefix;

const Nd2Files: React.FC = () => {
    const [nd2Files, setNd2Files] = useState<string[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [dialogMessage, setDialogMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        fetchND2Files();
    }, []);

    const fetchND2Files = async () => {
        try {
            const response = await axios.get<ListND2FilesResponse>(`${url_prefix}/cell_extraction/nd2_files`);
            setNd2Files(response.data.files);
        } catch (error) {
            console.error("Failed to fetch ND2 files", error);
        }
    };

    const handleNavigate = (fileName: string) => {
        navigate(`/cellextraction/?file_name=${fileName}`);
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
            setIsLoading(true);
            const formData = new FormData();
            formData.append("file", selectedFile);
            try {
                const response = await axios.post(`${url_prefix}/cell_extraction/nd2_files`, formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });
                setDialogMessage("File uploaded successfully!");
                setDialogOpen(true);
                setSelectedFile(null);
                fetchND2Files();
                console.log(response.data);
            } catch (error) {
                setDialogMessage("Failed to upload file.");
                setDialogOpen(true);
                console.error("Failed to upload file", error);
            } finally {
                setIsLoading(false);
            }
        }
    };

    const handleDelete = async (fileName: string) => {
        try {
            const response = await axios.delete(`${url_prefix}/cell_extraction/nd2_files/${fileName}`);
            setDialogMessage("File deleted successfully!");
            setDialogOpen(true);
            fetchND2Files();
            console.log(response.data);
        } catch (error) {
            setDialogMessage("Failed to delete file.");
            setDialogOpen(true);
            console.error("Failed to delete file", error);
        }
    };

    const handleCloseDialog = () => {
        setDialogOpen(false);
        window.location.reload();
    };

    const filteredFiles = nd2Files.filter(file => file.toLowerCase().includes(searchQuery.toLowerCase()));

    return (
        <Container>
            <Box>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Typography color="text.primary">ND2 files</Typography>
                </Breadcrumbs>
            </Box>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="space-between" mt={2}>
                <Grid container spacing={2} alignItems="center">
                    <Grid item xs={6}>
                        <TextField
                            label="Search ND2 files"
                            variant="outlined"
                            fullWidth
                            value={searchQuery}
                            onChange={handleSearchChange}
                            sx={{ height: '56px' }}
                        />
                    </Grid>
                    <Grid item xs={3}>
                        <input
                            accept=".nd2"
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
                                startIcon={<InsertDriveFileIcon />}
                                sx={{
                                    backgroundColor: 'background.paper',
                                    color: 'text.primary',
                                    width: '100%',
                                    height: '56px',
                                    textTransform: 'none',
                                    '&:hover': {
                                        backgroundColor: 'action.hover'
                                    }
                                }}
                            >
                                {selectedFile ? selectedFile.name : "Select ND2 file"}
                            </Button>
                        </label>
                    </Grid>
                    <Grid item xs={3}>
                        <Button
                            onClick={handleUpload}
                            variant="contained"
                            sx={{
                                backgroundColor: 'primary.main',
                                color: 'primary.contrastText',
                                width: '100%',
                                height: '56px',
                                '&:hover': {
                                    backgroundColor: 'primary.dark'
                                }
                            }}
                            startIcon={<FileUploadIcon />}
                            disabled={!selectedFile || isLoading}
                        >
                            Upload
                        </Button>
                    </Grid>
                </Grid>
            </Box>

            {isLoading ? (
                <Box display="flex" justifyContent="center" alignItems="center" mt={3}>
                    <CircularProgress />
                    <Typography variant="h6" ml={2}>Uploading the selected ND2 file...</Typography>
                </Box>
            ) : (
                <Box mt={3}>
                    <TableContainer component={Paper}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>ND2 Files</TableCell>
                                    <TableCell align="right">Actions</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {filteredFiles.map((file, index) => (
                                    <TableRow key={index}>
                                        <TableCell component="th" scope="row">
                                            {file}
                                        </TableCell>
                                        <TableCell align="right">
                                            <IconButton onClick={() => handleNavigate(file)}>
                                                <Typography>Extract cells </Typography>
                                                <NavigateNextIcon />
                                            </IconButton>
                                            {file !== "T256.nd2" && (
                                                <IconButton onClick={() => handleDelete(file)}>
                                                    <DeleteIcon color="error" />
                                                </IconButton>
                                            )}
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Box>
            )}

            <Dialog open={dialogOpen} onClose={handleCloseDialog}>
                <DialogTitle>{"File Operation Status"}</DialogTitle>
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

export default Nd2Files;
