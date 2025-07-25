import React, { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  Container,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Button,
  Grid,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Select,
  MenuItem,
  SelectChangeEvent,
  Link,
  Breadcrumbs,
  CircularProgress,
  TextField,
  Tooltip,
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import DatabaseIcon from "@mui/icons-material/Storage";
import FileUploadIcon from "@mui/icons-material/FileUpload";
import { useNavigate } from "react-router-dom";
import { useLocation } from "react-router-dom";
import TaskIcon from "@mui/icons-material/Task";
import DownloadIcon from "@mui/icons-material/Download";
import DriveFileMoveIcon from "@mui/icons-material/DriveFileMove";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";

interface ListDBResponse {
  databases: string[];
}

const url_prefix = settings.url_prefix;

const Databases: React.FC = () => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const defaultSearchWord = queryParams.get("default_search_word") ?? "";

  const [databases, setDatabases] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState(defaultSearchWord);

  // 表示モード
  const [displayMode, setDisplayMode] = useState(
    () => localStorage.getItem("displayMode") || "User uploaded"
  );

  // ファイルアップロード関連
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // ダイアログ関連
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [dialogMessage, setDialogMessage] = useState("");

  // “Mark as complete” 用の確認ダイアログ
  const [isConfirmDialogOpen, setIsConfirmDialogOpen] = useState(false);
  const [databaseToComplete, setDatabaseToComplete] = useState<string | null>(
    null
  );

  // アップロード済みDBが編集可能かチェックするマップ
  const [markableDatabases, setMarkableDatabases] = useState<{
    [key: string]: boolean;
  }>({});

  // プレビュー関連
  const [isPreviewDialogOpen, setIsPreviewDialogOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState("fluo");
  const [selectedLabel, setSelectedLabel] = useState("1");
  const [loadingPreview, setLoadingPreview] = useState(false);

  // メタデータ
  const [metadata, setMetadata] = useState<{ [key: string]: string }>({});
  const [newMetadata, setNewMetadata] = useState<{ [key: string]: string }>({});

  const navigate = useNavigate();

  /**
   * ユーザーアップロード or Completed or Validated用のデータを取得
   */
  const fetchDatabases = useCallback(async () => {
    try {
      const token = localStorage.getItem("access_token");
      const headers =
        token && displayMode !== "Validated"
          ? { Authorization: `Bearer ${token}` }
          : {};
      const response = await axios.get<ListDBResponse>(
        `${url_prefix}/databases`,
        { headers }
      );
      setDatabases(response.data.databases);

      // アップロード済みDBを抽出
      const uploadedDatabases = response.data.databases.filter((db) =>
        db.endsWith("-uploaded.db")
      );

      // アップロード済みDBが "Mark as complete" 可能かをチェック
      const markableStatus = await Promise.all(
        uploadedDatabases.map(async (db) => {
          try {
            const checkResponse = await axios.get(`${url_prefix}/databases/${db}`);
            return { db, markable: checkResponse.data };
          } catch (error) {
            console.error(`Failed to check markable status for ${db}`, error);
            return { db, markable: false };
          }
        })
      );

      const markableStatusMap = markableStatus.reduce(
        (acc, { db, markable }) => {
          acc[db] = markable;
          return acc;
        },
        {} as { [key: string]: boolean }
      );
      setMarkableDatabases(markableStatusMap);

      // 全データベース分のメタデータを取得
      const metadataResponses = await Promise.all(
        response.data.databases.map(async (db) => {
          try {
            const metadataResponse = await axios.get(
              `${url_prefix}/databases/${db}/metadata`
            );
            return { db, metadata: metadataResponse.data };
          } catch (error) {
            console.error(`Failed to fetch metadata for ${db}`, error);
            return { db, metadata: "" };
          }
        })
      );

      const metadataMap = metadataResponses.reduce(
        (acc, { db, metadata }) => {
          acc[db] = metadata;
          return acc;
        },
        {} as { [key: string]: string }
      );

      setMetadata(metadataMap);
      setNewMetadata(metadataMap);
    } catch (error) {
      console.error("Failed to fetch databases", error);
    }
  }, [displayMode]);

  /**
   * displayMode の変更に応じてデータを取得
   */
  useEffect(() => {
    fetchDatabases();
  }, [displayMode, fetchDatabases]);

  /**
   * 表示モードを切り替え
   */
  const handleDisplayModeChange = (event: SelectChangeEvent<string>) => {
    const newDisplayMode = event.target.value;
    setDisplayMode(newDisplayMode);
    localStorage.setItem("displayMode", newDisplayMode);
  };

  /**
   * ファイル入力変更
   */
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
    }
  };

  /**
   * 検索ワード入力変更
   */
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  /**
   * クリップボードにコピー
   */
  const handleCopyToClipboard = (dbName: string) => {
    navigator.clipboard
      .writeText(dbName)
      .then(() => {
        alert(`${dbName} copied to clipboard!`);
      })
      .catch((err) => {
        console.error("Failed to copy text: ", err);
      });
  };

  /**
   * DBファイルのアップロード
   */
  const handleUpload = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);
      try {
        await axios.post(`${url_prefix}/databases/upload`, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        setDialogMessage("Database uploaded successfully!");
        setIsDialogOpen(true);
        // アップロード後のリスト再取得
        fetchDatabases();
      } catch (error) {
        setDialogMessage("Failed to upload database.");
        setIsDialogOpen(true);
        console.error("Failed to upload database", error);
      }
    }
  };


  /**
   * メタデータを更新
   */
  const handleMetadataChange = async (dbName: string, updatedMetadata: string) => {
    setNewMetadata((prev) => ({
      ...prev,
      [dbName]: updatedMetadata,
    }));

    try {
      await axios.patch(
        `${url_prefix}/databases/${dbName}/update-metadata`,
        { metadata: updatedMetadata },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );
      setDialogMessage("Metadata updated successfully!");
      setIsDialogOpen(true);
      setMetadata((prev) => ({
        ...prev,
        [dbName]: updatedMetadata,
      }));
    } catch (error) {
      setDialogMessage("Failed to update metadata.");
      setIsDialogOpen(true);
      console.error("Failed to update metadata", error);
    }
  };

  /**
   * “Mark as Complete” ダイアログを開く
   */
  const handleOpenConfirmDialog = (database: string) => {
    setDatabaseToComplete(database);
    setIsConfirmDialogOpen(true);
  };

  /**
   * “Mark as Complete” ダイアログを閉じる
   */
  const handleCloseConfirmDialog = () => {
    setIsConfirmDialogOpen(false);
    setDatabaseToComplete(null);
  };

  /**
   * ユーザーがアップロードしたDBをCompleteにマーク
   */
  const handleMarkAsComplete = async () => {
    if (databaseToComplete) {
      try {
        const response = await axios.patch(
          `${url_prefix}/databases/${databaseToComplete}`
        );
        setDialogMessage(response.data.message);
        setIsDialogOpen(true);
        handleCloseConfirmDialog();
        // リストを再取得
        fetchDatabases();
      } catch (error) {
        setDialogMessage("Failed to mark database as complete.");
        setIsDialogOpen(true);
        console.error("Failed to mark database as complete", error);
      }
    }
  };

  /**
   * Completed DBをダウンロード
   */
  const handleDownload = async (database: string) => {
    try {
      const response = await axios.get(
        `${url_prefix}/databases/download-completed/${database}`,
        {
          responseType: "blob",
        }
      );

      const contentDisposition = response.headers["content-disposition"];
      const fileName = contentDisposition
        ? contentDisposition.split("filename=")[1]
        : database;

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", fileName);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      setDialogMessage("Failed to download database.");
      setIsDialogOpen(true);
      console.error("Failed to download database", error);
    }
  };

  /**
   * 画像プレビュー取得
   */
  const handlePreview = async (database: string) => {
    setLoadingPreview(true);
    try {
      const response = await axios.get(
        `${url_prefix}/databases/${database}/combined_images`,
        {
          params: {
            label: selectedLabel,
            mode: selectedMode,
          },
          responseType: "blob",
        }
      );

      const url = window.URL.createObjectURL(new Blob([response.data]));
      setPreviewImage(url);
      setIsPreviewDialogOpen(true);
    } catch (error) {
      setDialogMessage("Failed to fetch preview image.");
      setIsDialogOpen(true);
      console.error("Failed to fetch preview image", error);
    } finally {
      setLoadingPreview(false);
    }
  };

  /**
   * 指定のDBに画面遷移
   */
  const handleNavigate = (dbName: string) => {
    navigate(`/databases/?db_name=${dbName}`);
  };

  /**
   * ラベルソーターページへ遷移
   */
  const handleNavigateLabelSorter = (dbName: string) => {
    navigate(`/labelsorter?db_name=${dbName}`);
  };

  /**
   * 通常のダイアログを閉じる
   */
  const handleCloseDialog = () => {
    setIsDialogOpen(false);
  };

  /**
   * プレビュー用ダイアログを閉じる
   */
  const handleClosePreviewDialog = () => {
    setIsPreviewDialogOpen(false);
    setPreviewImage(null);
  };

  // displayMode と検索ワードに合わせてフィルタリング
  const filteredDatabases = databases.filter((database) => {
    const searchMatch = database.toLowerCase().includes(searchQuery.toLowerCase());

    if (displayMode === "User uploaded") {
      return searchMatch && database.endsWith("-uploaded.db");
    }
    if (displayMode === "Completed") {
      return searchMatch && database.endsWith("-completed.db");
    }
    if (displayMode === "Validated") {
      return (
        searchMatch &&
        !database.endsWith("-uploaded.db") &&
        !database.endsWith("-completed.db")
      );
    }
    return searchMatch;
  });


  return (
    <Container maxWidth={false} disableGutters>
      {/* パンくずリスト */}
      <Box>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Typography color="text.primary">Database Console</Typography>
        </Breadcrumbs>
      </Box>

      {/* 検索・モード選択・ファイルアップロードなど */}
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="space-between"
        mt={2}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={4}>
            <TextField
              label="Search Database"
              variant="outlined"
              fullWidth
              value={searchQuery}
              onChange={handleSearchChange}
              sx={{ height: "56px" }}
            />
          </Grid>

          <Grid item xs={2}>
            <Select
              value={displayMode}
              onChange={handleDisplayModeChange}
              displayEmpty
              inputProps={{ "aria-label": "Without label" }}
              fullWidth
              sx={{ height: "56px" }}
            >
              <MenuItem value="Validated">Validated</MenuItem>
              <MenuItem value="User uploaded">Uploaded</MenuItem>
              <MenuItem value="Completed">Completed</MenuItem>
            </Select>
          </Grid>

          <Grid item xs={3}>
            <input
              accept=".db"
              style={{ display: "none" }}
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
                  backgroundColor: 'background.paper',
                  color: 'text.primary',
                  width: '100%',
                  height: '56px',
                  textTransform: 'none',
                  '&:hover': {
                    backgroundColor: 'action.hover',
                  },
                }}
              >
                {selectedFile ? selectedFile.name : "Select Database"}
              </Button>
            </label>
          </Grid>

          {displayMode !== "Completed" && (
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
                    backgroundColor: 'primary.dark',
                  },
                }}
                startIcon={<FileUploadIcon />}
                disabled={!selectedFile}
              >
                Upload
              </Button>
            </Grid>
          )}

        </Grid>
      </Box>

      {/* テーブル表示部分 */}
      <Box mt={3}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Database Name</TableCell>
                <TableCell>Copy</TableCell>
                <TableCell align="center">Metadata</TableCell>
                {displayMode === "User uploaded" && (
                  <TableCell align="center" sx={{ whiteSpace: "nowrap" }}>
                    Mark as Complete
                  </TableCell>
                )}
                {displayMode === "Completed" && (
                  <TableCell align="center">Export</TableCell>
                )}

                <TableCell
                  align="center"
                  sx={{ whiteSpace: "nowrap" }}
                >
                  <Box
                    display="flex"
                    justifyContent="center"
                    alignItems="center"
                    flexWrap="nowrap"
                  >
                    <Box>
                      <Typography>Mode</Typography>
                    </Box>
                    <Box ml={1} display="flex" flexWrap="nowrap">
                      <Select
                        value={selectedMode}
                        onChange={(e) => setSelectedMode(e.target.value)}
                        displayEmpty
                        inputProps={{ "aria-label": "Without label" }}
                        sx={{ marginRight: 1, height: "25px" }}
                      >
                        <MenuItem value="fluo">Fluo</MenuItem>
                        <MenuItem value="ph">Ph</MenuItem>
                        <MenuItem value="ph_contour">Ph + contour</MenuItem>
                        <MenuItem value="fluo_contour">Fluo + contour</MenuItem>
                        <MenuItem value="fluo2">fluo2</MenuItem>
                        <MenuItem value="fluo2_contour">fluo2 + contour</MenuItem>
                        <MenuItem value="replot_fluo1">replot fluo1</MenuItem>
                        <MenuItem value="replot_fluo2">replot fluo2</MenuItem>
                      </Select>
                      <Select
                        value={selectedLabel}
                        onChange={(e) => setSelectedLabel(e.target.value)}
                        displayEmpty
                        inputProps={{ "aria-label": "Without label" }}
                        sx={{ marginRight: 1, height: "25px" }}
                      >
                        <MenuItem value="N/A">N/A</MenuItem>
                        <MenuItem value="1">1</MenuItem>
                        <MenuItem value="2">2</MenuItem>
                        <MenuItem value="3">3</MenuItem>
                      </Select>
                    </Box>
                  </Box>
                </TableCell>

                <TableCell align="center">Access</TableCell>
                <TableCell align="center">Sort labels</TableCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {filteredDatabases.map((database, index) => (
                  <TableRow key={index}>
                      <TableCell component="th" scope="row">
                        <Tooltip title={database} placement="top">
                          <Typography noWrap>
                            {database.length > 30
                              ? `${database.substring(0, 30)}...`
                              : database}
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

                      {/* メタデータ編集欄 */}
                      <TableCell>
                        <Box display="flex" alignItems="center" justifyContent="center">
                          <TextField
                            value={newMetadata[database] || ""}
                            onChange={(e) =>
                              setNewMetadata((prev) => ({
                                ...prev,
                                [database]: e.target.value,
                              }))
                            }
                            onBlur={() =>
                              handleMetadataChange(
                                database,
                                newMetadata[database] || ""
                              )
                            }
                            fullWidth
                            placeholder="e.g., yyyy/mm/dd"
                            InputProps={{
                              sx: {
                                height: "40px",
                                padding: "0",
                              },
                              autoComplete: "off",
                            }}
                          />
                        </Box>
                      </TableCell>

                      {/* User uploaded: Mark as Complete */}
                      {displayMode === "User uploaded" && (
                        <TableCell align="center" sx={{ whiteSpace: "nowrap" }}>
                          <Button
                            variant="contained"
                            sx={{
                              backgroundColor: markableDatabases[database]
                                ? "green"
                                : "grey",
                              color: "white",
                              whiteSpace: "nowrap",
                              "&:hover": {
                                backgroundColor: markableDatabases[database]
                                  ? "darkgreen"
                                  : "grey",
                              },
                            }}
                            onClick={() => handleOpenConfirmDialog(database)}
                            startIcon={<TaskIcon />}
                            disabled={!markableDatabases[database]}
                          >
                            Mark as Complete
                          </Button>
                        </TableCell>
                      )}

                      {/* Completed: ダウンロード */}
                      {displayMode === "Completed" && (
                        <TableCell align="center">
                          <Button
                            variant="contained"
                            sx={{
                              backgroundColor: 'primary.main',
                              color: 'primary.contrastText',
                              '&:hover': {
                                backgroundColor: 'primary.dark',
                              },
                            }}
                            onClick={() => handleDownload(database)}
                            startIcon={<DownloadIcon />}
                          >
                            Export
                          </Button>
                        </TableCell>
                      )}

                      {/* プレビューとDBアクセスボタン */}
                      <>
                        <TableCell align="center">
                          <Button
                            variant="contained"
                            sx={{
                              backgroundColor: 'primary.main',
                              color: 'primary.contrastText',
                              textTransform: 'none',
                              '&:hover': {
                                backgroundColor: 'primary.dark',
                              },
                            }}
                          onClick={() => handlePreview(database)}
                        >
                          Preview
                          </Button>
                        </TableCell>
                        <TableCell align="right">
                          <IconButton onClick={() => handleNavigate(database)}>
                            <Typography>Access database </Typography>
                            <NavigateNextIcon />
                          </IconButton>
                        </TableCell>
                        <TableCell align="right">
                          <IconButton onClick={() => handleNavigateLabelSorter(database)}>
                            <Typography>Sort labels </Typography>
                            <NavigateNextIcon />
                          </IconButton>
                        </TableCell>
                      </>
                    </TableRow>
                  ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>

      {/* 通常メッセージダイアログ */}
      <Dialog open={isDialogOpen} onClose={handleCloseDialog}>
        <DialogTitle>{"Status"}</DialogTitle>
        <DialogContent>
          <DialogContentText>{dialogMessage}</DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Mark as Complete用 ダイアログ */}
      <Dialog open={isConfirmDialogOpen} onClose={handleCloseConfirmDialog}>
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

      {/* プレビュー用ダイアログ */}
      <Dialog open={isPreviewDialogOpen} onClose={handleClosePreviewDialog}>
        <DialogTitle>{"Preview Image"}</DialogTitle>
        <DialogContent>
          {loadingPreview ? (
            <Box display="flex" justifyContent="center">
              <CircularProgress />
            </Box>
          ) : (
            previewImage && (
              <img src={previewImage} alt="Preview" style={{ width: "100%" }} />
            )
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
