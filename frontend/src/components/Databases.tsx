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
import DriveFileMoveIcon from '@mui/icons-material/DriveFileMove';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

interface ListDBResponse {
    databases: string[];
}

const url_prefix = settings.url_prefix;

const Databases: React.FC = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const default_search_word = queryParams.get('default_search_word') ?? "";
    const [databases, setDatabases] = useState<string[]>([]);
    const [dropboxFiles, setDropboxFiles] = useState<string[]>([]);
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

        const fetchDropboxFiles = async () => {
            try {
                const response = await axios.get(`${url_prefix}/dropbox/list_databases`);
                setDropboxFiles(response.data.files);
            } catch (error) {
                console.error("Failed to fetch Dropbox files", error);
            }
        };

        if (displayMode === 'Dropbox') {
            fetchDropboxFiles();
        } else {
            fetchDatabases();
        }
    }, [displayMode]);

    // @router_dropbox.get("/download")
    // async def download_file(file_name: str):
    // return { "message": await DropboxCrud().download_file(file_name) }

    const handleDropboxDownload = async (file: string) => {
        try {
            const response = await axios.get(`${url_prefix}/dropbox/download/${file}`, {
                responseType: 'blob'
            });

            const contentDisposition = response.headers['content-disposition'];
            const fileName = contentDisposition ? contentDisposition.split('filename=')[1] : file;

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
    }

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

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchQuery(event.target.value);
    };

    // Define the filtered data depending on the display mode
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

    const filteredDropboxFiles = dropboxFiles.filter(file => file.toLowerCase().includes(searchQuery.toLowerCase()));

    const handleCopyToClipboard = (dbName: string) => {
        navigator.clipboard.writeText(dbName).then(() => {
            alert(`${dbName} copied to clipboard!`);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
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

    const handleBackup = async () => {
        try {
            const response = await axios.post(`${url_prefix}/dropbox/databases/backup`);
            setDialogMessage("Backup completed successfully!");
            setDialogOpen(true);
        } catch (error) {
            setDialogMessage("Failed to backup the database.");
            setDialogOpen(true);
            console.error("Backup failed", error);
        }
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

    const handleNavigate = (dbName: string) => {
        navigate(`/databases/?db_name=${dbName}`);
    };

    const handleCloseDialog = () => {
        setDialogOpen(false);
    };

    const handleClosePreviewDialog = () => {
        setPreviewDialogOpen(false);
        setPreviewImage(null);
    };


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
                            <MenuItem value="Dropbox">Dropbox</MenuItem> {/* New Dropbox mode */}
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
                                    textTransform: 'none',
                                    '&:hover': {
                                        backgroundColor: 'lightgrey'
                                    }
                                }}
                            >
                                {selectedFile ? selectedFile.name : "Select Database"}
                            </Button>
                        </label>
                    </Grid>
                    {displayMode !== 'Completed' && (<Grid item xs={3}>
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
                    </Grid>)}

                    {displayMode === 'Completed' && (
                        <Grid item xs={3}>
                            <Button
                                onClick={handleBackup}
                                variant="contained"
                                sx={{
                                    backgroundColor: '#0061FE',
                                    color: 'white',
                                    width: '100%',
                                    height: '56px',
                                    textTransform: 'none',
                                    '&:hover': {
                                        backgroundColor: 'grey'
                                    }
                                }}
                                startIcon={<DriveFileMoveIcon />}
                            >
                                Sync to Dropbox
                            </Button>
                        </Grid>
                    )}
                </Grid>
            </Box>

            <Box mt={3}>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Database Name</TableCell>
                                <TableCell>Copy</TableCell>
                                {displayMode === 'User uploaded' && <TableCell align="center">Mark as Complete</TableCell>}
                                {displayMode === 'Completed' && <TableCell align="center">Export</TableCell>}
                                {displayMode !== 'Dropbox' && (
                                    <>
                                        <TableCell align="center">Metadata</TableCell>
                                        <TableCell align="center">
                                            <Box display="flex" justifyContent="center" alignItems="center">
                                                <Box>
                                                    <Typography>Mode</Typography>
                                                </Box>
                                                <Box ml={1}>
                                                    <Select
                                                        value={selectedMode}
                                                        onChange={(e) => setSelectedMode(e.target.value)}
                                                        displayEmpty
                                                        inputProps={{ 'aria-label': 'Without label' }}
                                                        sx={{ marginRight: 1, height: '25px' }}
                                                    >
                                                        <MenuItem value="fluo">Fluo</MenuItem>
                                                        <MenuItem value="ph">Ph</MenuItem>
                                                        <MenuItem value="ph_contour">Ph + contour</MenuItem>
                                                        <MenuItem value="fluo_contour">Fluo + contour</MenuItem>
                                                    </Select>
                                                    <Select
                                                        value={selectedLabel}
                                                        onChange={(e) => setSelectedLabel(e.target.value)}
                                                        displayEmpty
                                                        inputProps={{ 'aria-label': 'Without label' }}
                                                        sx={{ marginRight: 1, height: '25px' }}
                                                    >
                                                        <MenuItem value="N/A">N/A</MenuItem>
                                                        <MenuItem value="1">1</MenuItem>
                                                        <MenuItem value="2">2</MenuItem>
                                                        <MenuItem value="3">3</MenuItem>
                                                    </Select>
                                                </Box>
                                            </Box>
                                        </TableCell>
                                        <TableCell align="center"></TableCell>
                                    </>)}
                                {displayMode === 'Dropbox' && (
                                    <TableCell align="center">Download</TableCell>
                                )}
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {displayMode === 'Dropbox'
                                ? filteredDropboxFiles.map((file, index) => (
                                    <TableRow key={index}>
                                        <TableCell component="th" scope="row">
                                            <Tooltip title={file} placement="top">
                                                <Typography noWrap>
                                                    {
                                                        displayMode === 'Dropbox' && (file.length > 30 ? `${file.substring(0, 30)}...` : file)
                                                    }
                                                    {
                                                        displayMode !== 'Dropbox' && (file.length > 15 ? `${file.substring(0, 15)}...` : file)
                                                    }

                                                </Typography>
                                            </Tooltip>
                                        </TableCell>
                                        <TableCell>
                                            <Tooltip title="Copy to clipboard">
                                                <IconButton onClick={() => handleCopyToClipboard(file)}>
                                                    <ContentCopyIcon />
                                                </IconButton>
                                            </Tooltip>
                                        </TableCell>
                                        {
                                            displayMode === 'Dropbox' && (
                                                <TableCell align="center">
                                                    <Button
                                                        variant="contained"
                                                        sx={{
                                                            backgroundColor: 'black',
                                                            color: 'white',
                                                            '&:hover': {
                                                                backgroundColor: 'gray'
                                                            }
                                                        }}
                                                        onClick={() => handleDropboxDownload(file)}
                                                        startIcon={<DownloadIcon />}
                                                    >
                                                        Download
                                                    </Button>
                                                </TableCell>
                                            )
                                        }
                                        {displayMode !== 'Dropbox' && (<TableCell align="center">N/A</TableCell>)}
                                    </TableRow>
                                ))
                                : filteredDatabases.map((database, index) => (
                                    <TableRow key={index}>
                                        <TableCell component="th" scope="row">
                                            <Tooltip title={database} placement="top">
                                                <Typography noWrap>
                                                    {database.length > 15 ? `${database.substring(0, 15)}...` : database}
                                                </Typography>
                                            </Tooltip>
                                        </TableCell>
                                        <TableCell>
                                            <Tooltip title="Copy to clipboard">
                                                <IconButton onClick={() => handleCopyToClipboard(database)}>
                                                    <ContentCopyIcon />
                                                </IconButton>
                                            </Tooltip>
                                        </TableCell>
                                        <TableCell>
                                            <Box display="flex" alignItems="center" justifyContent="center" height="100%">
                                                <TextField
                                                    value={newMetadata[database] || ""}
                                                    onChange={(e) => setNewMetadata(prevMetadata => ({
                                                        ...prevMetadata,
                                                        [database]: e.target.value
                                                    }))}
                                                    onBlur={() => handleMetadataChange(database, newMetadata[database] || "")}
                                                    fullWidth
                                                    placeholder="enter details"
                                                    InputProps={{
                                                        sx: {
                                                            height: '40px',
                                                            padding: '0',
                                                        },
                                                        autoComplete: 'off'
                                                    }}
                                                />
                                            </Box>
                                        </TableCell>
                                        {displayMode === 'User uploaded' && (
                                            <TableCell align="center">
                                                <Button
                                                    variant="contained"
                                                    sx={{
                                                        backgroundColor: markableDatabases[database] ? 'green' : 'grey',
                                                        color: 'white',
                                                        '&:hover': {
                                                            backgroundColor: markableDatabases[database] ? 'darkgreen' : 'grey'
                                                        }
                                                    }}
                                                    onClick={() => handleOpenConfirmDialog(database)}
                                                    startIcon={<TaskIcon />}
                                                    disabled={!markableDatabases[database]}
                                                >
                                                    Mark as Complete
                                                </Button>
                                            </TableCell>
                                        )}
                                        {displayMode === 'Completed' && (
                                            <TableCell align="center">
                                                <Button
                                                    variant="contained"
                                                    sx={{
                                                        backgroundColor: 'black',
                                                        color: 'white',
                                                        '&:hover': {
                                                            backgroundColor: 'gray'
                                                        }
                                                    }}
                                                    onClick={() => handleDownload(database)}
                                                    startIcon={<DownloadIcon />}
                                                >
                                                    Export
                                                </Button>
                                            </TableCell>
                                        )}
                                        {displayMode !== 'Dropbox' && (
                                            <>
                                                <TableCell align="center">
                                                    <Button
                                                        variant="contained"
                                                        sx={{
                                                            backgroundColor: 'black',
                                                            color: 'white',
                                                            textTransform: 'none',
                                                            '&:hover': {
                                                                backgroundColor: 'gray'
                                                            }
                                                        }}
                                                        onClick={() => handlePreview(database)}
                                                    >
                                                        Export Preview
                                                    </Button>
                                                </TableCell>
                                                <TableCell align="right">
                                                    <IconButton onClick={() => handleNavigate(database)}>
                                                        <Typography>Access database </Typography>
                                                        <NavigateNextIcon />
                                                    </IconButton>
                                                </TableCell>
                                            </>
                                        )}
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

            <Dialog open={previewDialogOpen} onClose={handleClosePreviewDialog}>
                <DialogTitle>{"Preview Image"}</DialogTitle>
                <DialogContent>
                    {loadingPreview ? (
                        <Box display="flex" justifyContent="center">
                            <CircularProgress />
                        </Box>
                    ) : (
                        previewImage && <img src={previewImage} alt="Preview" style={{ width: '100%' }} />
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleClosePreviewDialog} color="primary">
                        Close
                    </Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
};

export default Databases;
