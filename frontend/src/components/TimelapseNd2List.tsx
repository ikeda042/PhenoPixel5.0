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

const TimelapseNd2List: React.FC = () => {
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

    /**
     * ND2ファイルの一覧を取得
     * GET /tlengine/nd2_files
     */
    const fetchND2Files = async () => {
        try {
            // ---- エンドポイントを /tlengine/nd2_files に修正 ----
            const response = await axios.get<ListND2FilesResponse>(`${url_prefix}/tlengine/nd2_files`);
            setNd2Files(response.data.files);
        } catch (error) {
            console.error("Failed to fetch ND2 files", error);
        }
    };

    /**
     * 詳細ページ（パース画面）へ遷移
     * ここでは /tlparser/?file_name=○○○ へ飛ばしている想定
     */
    const handleNavigate = (fileName: string) => {
        navigate(`/tlparser/?file_name=${fileName}`);
    };

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchQuery(event.target.value);
    };

    /**
     * ファイル選択
     */
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            setSelectedFile(event.target.files[0]);
        }
    };

    /**
     * ND2ファイルをアップロード
     * POST /tlengine/nd2_files
     */
    const handleUpload = async () => {
        if (selectedFile) {
            setIsLoading(true);
            const formData = new FormData();
            formData.append("file", selectedFile);
            try {
                // ---- エンドポイントを /tlengine/nd2_files に修正 ----
                const response = await axios.post(
                    `${url_prefix}/tlengine/nd2_files`,
                    formData,
                    {
                        headers: {
                            'Content-Type': 'multipart/form-data'
                        }
                    }
                );
                setDialogMessage("File uploaded successfully!");
                setDialogOpen(true);
                setSelectedFile(null);
                fetchND2Files(); // 一覧を再取得
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

    /**
     * ND2ファイルを削除
     * DELETE /tlengine/nd2_files?file_path=uploaded_files/xxx_timelapse.nd2
     */
    const handleDelete = async (fileName: string) => {
        try {
            setIsLoading(true);
            // ---- DELETE 時はパスパラメータではなくクエリパラメータを利用 ----
            // 例: /tlengine/nd2_files?file_path=uploaded_files/xxx_timelapse.nd2
            const response = await axios.delete(`${url_prefix}/tlengine/nd2_files`, {
                params: { file_path: `uploaded_files/${fileName}` }
            });
            setDialogMessage("File deleted successfully!");
            setDialogOpen(true);
            fetchND2Files();
            console.log(response.data);
        } catch (error) {
            setDialogMessage("Failed to delete file.");
            setDialogOpen(true);
            console.error("Failed to delete file", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleCloseDialog = () => {
        setDialogOpen(false);
        // 必要に応じて画面リロード
        // window.location.reload();
    };

    // フィルター
    const filteredFiles = nd2Files.filter(file =>
        file.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <Container>
            <Box>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Typography color="text.primary">Timelapse ND2 files</Typography>
                </Breadcrumbs>
            </Box>

            {/* ファイル検索 & アップロード */}
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="space-between" mt={2}>
                <Grid container spacing={2} alignItems="center">
                    {/* 検索 */}
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
                    {/* ファイル選択 */}
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
                                {selectedFile ? selectedFile.name : "Select Timelapse ND2 file"}
                            </Button>
                        </label>
                    </Grid>
                    {/* アップロード */}
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
                            disabled={!selectedFile || isLoading}
                        >
                            Upload
                        </Button>
                    </Grid>
                </Grid>
            </Box>

            {/* ローディング中表示 */}
            {isLoading ? (
                <Box display="flex" justifyContent="center" alignItems="center" mt={3}>
                    <CircularProgress />
                    <Typography variant="h6" ml={2}>Processing...</Typography>
                </Box>
            ) : (
                // ND2ファイル一覧
                <Box mt={3}>
                    <TableContainer component={Paper}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell><b>Timelapse ND2 Files</b></TableCell>
                                    <TableCell align="right"></TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {filteredFiles.map((file, index) => (
                                    <TableRow key={index}>
                                        <TableCell component="th" scope="row">
                                            {file}
                                        </TableCell>
                                        <TableCell align="right">
                                            {/* パース画面へ移動 */}
                                            <IconButton onClick={() => handleNavigate(file)}>
                                                <Typography>Parse TLND2</Typography>
                                                <NavigateNextIcon />
                                            </IconButton>
                                            {/* 削除 */}
                                            <IconButton onClick={() => handleDelete(file)}>
                                                <DeleteIcon color="error" />
                                            </IconButton>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Box>
            )}

            {/* 操作結果のダイアログ */}
            <Dialog open={dialogOpen} onClose={handleCloseDialog}>
                <DialogTitle>{"File Operation Status"}</DialogTitle>
                <DialogContent>
                    <DialogContentText>{dialogMessage}</DialogContentText>
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

export default TimelapseNd2List;
