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
import { useNavigate, useLocation } from "react-router-dom";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import DatabaseIcon from "@mui/icons-material/Storage";
import FileUploadIcon from "@mui/icons-material/FileUpload";
import TaskIcon from "@mui/icons-material/Task";
import DownloadIcon from "@mui/icons-material/Download";
import DriveFileMoveIcon from "@mui/icons-material/DriveFileMove";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";

interface ListDBResponse {
  databases: string[];
}

const url_prefix = settings.url_prefix;

const Databases: React.FC = () => {
  // ------------------------------------------------------------
  // State Hooks
  // ------------------------------------------------------------
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const default_search_word = queryParams.get("default_search_word") ?? "";
  const [databases, setDatabases] = useState<string[]>([]);
  const [dropboxFiles, setDropboxFiles] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState(default_search_word);
  const [displayMode, setDisplayMode] = useState(
    () => localStorage.getItem("displayMode") || "User uploaded"
  );
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // ダイアログの開閉とメッセージ関連
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogMessage, setDialogMessage] = useState("");

  // 完了確認ダイアログ関連
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [databaseToComplete, setDatabaseToComplete] = useState<string | null>(
    null
  );

  // 「Mark as Complete」ボタンを押せるかどうかの判定 (APIで取得したbool値を格納)
  const [markableDatabases, setMarkableDatabases] = useState<{
    [key: string]: boolean;
  }>({});

  // プレビュー画像用ダイアログ
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState("fluo");
  const [selectedLabel, setSelectedLabel] = useState("1");
  const [loadingPreview, setLoadingPreview] = useState(false);

  // メタデータ関連
  const [metadata, setMetadata] = useState<{ [key: string]: string }>({});
  const [newMetadata, setNewMetadata] = useState<{ [key: string]: string }>({});

  const navigate = useNavigate();

  // ------------------------------------------------------------
  // データベース一覧を取得する非同期関数 (再利用のため useCallback で定義)
  // ------------------------------------------------------------
  const fetchDatabases = useCallback(async () => {
    try {
      // データベース一覧の取得
      const response = await axios.get<ListDBResponse>(`${url_prefix}/databases`);
      setDatabases(response.data.databases);

      // アップロード済み(DB名が“-uploaded.db”で終わる)のものだけ抽出
      const uploadedDatabases = response.data.databases.filter((db) =>
        db.endsWith("-uploaded.db")
      );

      // 各アップロードDBが「Mark as Complete」できるかどうか取得
      const markableStatus = await Promise.all(
        uploadedDatabases.map(async (db) => {
          const checkResponse = await axios.get(`${url_prefix}/databases/${db}`);
          return { db, markable: checkResponse.data };
        })
      );

      // markableStatusを { dbName: true/false } の形式に整形
      const markableStatusMap = markableStatus.reduce(
        (acc, { db, markable }) => {
          acc[db] = markable;
          return acc;
        },
        {} as { [key: string]: boolean }
      );
      setMarkableDatabases(markableStatusMap);

      // 各DBのメタデータをまとめて取得
      const metadataResponses = await Promise.all(
        response.data.databases.map(async (db) => {
          const metadataResponse = await axios.get(
            `${url_prefix}/databases/${db}/metadata`
          );
          return { db, metadata: metadataResponse.data };
        })
      );

      // メタデータを { dbName: メタデータ文字列 } の形に変換
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
  }, []);

  // ------------------------------------------------------------
  // DropboxにあるDBファイル一覧を取得する非同期関数
  // ------------------------------------------------------------
  const fetchDropboxFiles = useCallback(async () => {
    try {
      const response = await axios.get(`${url_prefix}/dropbox/list_databases`);
      setDropboxFiles(response.data.files);
    } catch (error) {
      console.error("Failed to fetch Dropbox files", error);
    }
  }, []);

  // ------------------------------------------------------------
  // useEffect
  // displayModeが変わるたびにデータ取得 (Dropbox or DB一覧)
  // ------------------------------------------------------------
  useEffect(() => {
    if (displayMode === "Dropbox") {
      fetchDropboxFiles();
    } else {
      fetchDatabases();
    }
  }, [displayMode, fetchDatabases, fetchDropboxFiles]);

  // ------------------------------------------------------------
  // イベントハンドラ
  // ------------------------------------------------------------

  // 検索欄のテキスト変更
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  // 表示モード(Validated / User uploaded / Completed / Dropbox)の変更
  const handleDisplayModeChange = (event: SelectChangeEvent<string>) => {
    const newDisplayMode = event.target.value;
    setDisplayMode(newDisplayMode);
    localStorage.setItem("displayMode", newDisplayMode);
  };

  // ファイル選択時
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setSelectedFile(event.target.files[0]);
    }
  };

  // DBファイルをアップロード
  const handleUpload = async () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);
      try {
        const response = await axios.post(`${url_prefix}/databases/upload`, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        setDialogMessage("Database uploaded successfully!");
        setDialogOpen(true);
        console.log(response.data);
        // アップロード後、一覧を再取得
        fetchDatabases();
      } catch (error) {
        setDialogMessage("Failed to upload database.");
        setDialogOpen(true);
        console.error("Failed to upload database", error);
      }
    }
  };

  // ユーザーが「Mark as Complete」ボタンを押したときに呼ばれる
  // 確認用ダイアログを開く
  const handleOpenConfirmDialog = (database: string) => {
    setDatabaseToComplete(database);
    setConfirmDialogOpen(true);
  };

  // 確認用ダイアログを閉じる
  const handleCloseConfirmDialog = () => {
    setConfirmDialogOpen(false);
    setDatabaseToComplete(null);
  };

  // 実際に「Mark as Complete」を確定する
  const handleMarkAsComplete = async () => {
    if (!databaseToComplete) return;
    try {
      const response = await axios.patch(
        `${url_prefix}/databases/${databaseToComplete}`
      );
      setDialogMessage(response.data.message);
      setDialogOpen(true);
      // 確認ダイアログを閉じる
      handleCloseConfirmDialog();
      // ======== 【バグ修正ポイント】 ========
      // 以前は window.location.reload() があり、
      // それにより二重でモーダルが出ることがありました。
      // ここでは代わりに fetchDatabases() を呼び出して再取得し、
      // リロードはしないようにします。
      // ================================
      fetchDatabases();
    } catch (error) {
      setDialogMessage("Failed to mark database as complete.");
      setDialogOpen(true);
      console.error("Failed to mark database as complete", error);
    }
  };

  // Completed DBのダウンロード
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
      setDialogOpen(true);
      console.error("Failed to download database", error);
    }
  };

  // Dropbox上のファイルをダウンロード
  const handleDropboxDownload = async (file: string) => {
    try {
      await axios.get(`${url_prefix}/dropbox/download/${file}`, {
        responseType: "blob",
      });
      // 成功メッセージは不要であれば省略可能
      setDialogMessage("Downloaded from Dropbox successfully!");
      setDialogOpen(true);
    } catch (error) {
      setDialogMessage("Failed to download database.");
      setDialogOpen(true);
      console.error("Failed to download database", error);
    }
  };

  // 「Sync to Dropbox」ボタン押下時 (Completed のバックアップ)
  const handleBackup = async () => {
    try {
      await axios.post(`${url_prefix}/dropbox/databases/backup`);
      setDialogMessage("Backup completed successfully!");
      setDialogOpen(true);
    } catch (error) {
      setDialogMessage("Failed to backup the database.");
      setDialogOpen(true);
      console.error("Backup failed", error);
    }
  };

  // メタデータの入力フィールドがフォーカスを外れたタイミングで更新
  const handleMetadataChange = async (dbName: string, newMetadataValue: string) => {
    setNewMetadata((prev) => ({
      ...prev,
      [dbName]: newMetadataValue,
    }));

    try {
      await axios.patch(
        `${url_prefix}/databases/${dbName}/update-metadata`,
        { metadata: newMetadataValue },
        { headers: { "Content-Type": "application/json" } }
      );
      setDialogMessage("Metadata updated successfully!");
      setDialogOpen(true);
      setMetadata((prev) => ({
        ...prev,
        [dbName]: newMetadataValue,
      }));
    } catch (error) {
      setDialogMessage("Failed to update metadata.");
      setDialogOpen(true);
      console.error("Failed to update metadata", error);
    }
  };

  // プレビュー画像の取得
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
      setPreviewDialogOpen(true);
    } catch (error) {
      setDialogMessage("Failed to fetch preview image.");
      setDialogOpen(true);
      console.error("Failed to fetch preview image", error);
    } finally {
      setLoadingPreview(false);
    }
  };

  // データベースアクセス画面へ遷移
  const handleNavigate = (dbName: string) => {
    navigate(`/databases/?db_name=${dbName}`);
  };

  // ダイアログを閉じる
  const handleCloseDialog = () => {
    setDialogOpen(false);
  };

  // プレビューダイアログを閉じる
  const handleClosePreviewDialog = () => {
    setPreviewDialogOpen(false);
    setPreviewImage(null);
  };

  // ------------------------------------------------------------
  // フィルタリング処理
  // ------------------------------------------------------------
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

  const filteredDropboxFiles = dropboxFiles.filter((file) =>
    file.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // ------------------------------------------------------------
  // クリップボードにDB名をコピーする
  // ------------------------------------------------------------
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

  // ------------------------------------------------------------
  // レンダリング
  // ------------------------------------------------------------
  return (
    <Container>
      {/* パンくずリスト */}
      <Box>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Typography color="text.primary">Database Console</Typography>
        </Breadcrumbs>
      </Box>

      {/* 検索バー、ドロップダウン、ファイルアップロードUI */}
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="space-between"
        mt={2}
      >
        <Grid container spacing={2} alignItems="center">
          {/* 検索テキストフィールド */}
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

          {/* 表示モードを選択するSelect */}
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
              <MenuItem value="Dropbox">Dropbox</MenuItem>
            </Select>
          </Grid>

          {/* ファイル選択ボタン */}
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
                  backgroundColor: "white",
                  color: "black",
                  width: "100%",
                  height: "56px",
                  textTransform: "none",
                  "&:hover": {
                    backgroundColor: "lightgrey",
                  },
                }}
              >
                {selectedFile ? selectedFile.name : "Select Database"}
              </Button>
            </label>
          </Grid>

          {/* アップロードボタン (Completedモード時は非表示) */}
          {displayMode !== "Completed" && (
            <Grid item xs={3}>
              <Button
                onClick={handleUpload}
                variant="contained"
                sx={{
                  backgroundColor: "black",
                  color: "white",
                  width: "100%",
                  height: "56px",
                  "&:hover": {
                    backgroundColor: "grey",
                  },
                }}
                startIcon={<FileUploadIcon />}
                disabled={!selectedFile}
              >
                Upload
              </Button>
            </Grid>
          )}

          {/* Completed時のみ表示するバックアップボタン */}
          {displayMode === "Completed" && (
            <Grid item xs={3}>
              <Button
                onClick={handleBackup}
                variant="contained"
                sx={{
                  backgroundColor: "#0061FE",
                  color: "white",
                  width: "100%",
                  height: "56px",
                  textTransform: "none",
                  "&:hover": {
                    backgroundColor: "grey",
                  },
                }}
                startIcon={<DriveFileMoveIcon />}
              >
                Sync to Dropbox
              </Button>
            </Grid>
          )}
        </Grid>
      </Box>

      {/* テーブル表示部 */}
      <Box mt={3}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Database Name</TableCell>
                <TableCell>Copy</TableCell>
                {displayMode !== "Dropbox" && (
                  <>
                    <TableCell align="center">Metadata</TableCell>
                    {displayMode === "User uploaded" && (
                      <TableCell align="center">Mark as Complete</TableCell>
                    )}
                    {displayMode === "Completed" && (
                      <TableCell align="center">Export</TableCell>
                    )}
                    {/* プレビューモード＆ラベル選択 */}
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
                            inputProps={{ "aria-label": "Without label" }}
                            sx={{ marginRight: 1, height: "25px" }}
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
                    <TableCell align="center"></TableCell>
                  </>
                )}
                {displayMode === "Dropbox" && (
                  <TableCell align="center">Download</TableCell>
                )}
              </TableRow>
            </TableHead>
            <TableBody>
              {displayMode === "Dropbox"
                ? filteredDropboxFiles.map((file, index) => (
                    <TableRow key={index}>
                      <TableCell component="th" scope="row">
                        <Tooltip title={file} placement="top">
                          <Typography noWrap>
                            {file.length > 30 ? `${file.substring(0, 30)}...` : file}
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
                      <TableCell align="center">
                        <Button
                          variant="contained"
                          sx={{
                            backgroundColor: "black",
                            color: "white",
                            "&:hover": {
                              backgroundColor: "gray",
                            },
                          }}
                          onClick={() => handleDropboxDownload(file)}
                          startIcon={<DownloadIcon />}
                        >
                          Download
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                : filteredDatabases.map((database, index) => (
                    <TableRow key={index}>
                      <TableCell component="th" scope="row">
                        <Tooltip title={database} placement="top">
                          <Typography noWrap>
                            {database.length > 15
                              ? `${database.substring(0, 15)}...`
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

                      {/* Metadata */}
                      <TableCell>
                        <Box
                          display="flex"
                          alignItems="center"
                          justifyContent="center"
                          height="100%"
                        >
                          <TextField
                            value={newMetadata[database] || ""}
                            onChange={(e) =>
                              setNewMetadata((prev) => ({
                                ...prev,
                                [database]: e.target.value,
                              }))
                            }
                            // フォーカスを外したらメタデータ更新
                            onBlur={() =>
                              handleMetadataChange(database, newMetadata[database] || "")
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

                      {/* Mark as Complete (Uploadedモードのみ) */}
                      {displayMode === "User uploaded" && (
                        <TableCell align="center">
                          <Button
                            variant="contained"
                            sx={{
                              backgroundColor: markableDatabases[database]
                                ? "green"
                                : "grey",
                              color: "white",
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

                      {/* CompletedモードのExportボタン */}
                      {displayMode === "Completed" && (
                        <TableCell align="center">
                          <Button
                            variant="contained"
                            sx={{
                              backgroundColor: "black",
                              color: "white",
                              "&:hover": {
                                backgroundColor: "gray",
                              },
                            }}
                            onClick={() => handleDownload(database)}
                            startIcon={<DownloadIcon />}
                          >
                            Export
                          </Button>
                        </TableCell>
                      )}

                      {/* プレビューボタン (Dropbox以外で共通) & Database画面遷移ボタン */}
                      <>
                        <TableCell align="center">
                          <Button
                            variant="contained"
                            sx={{
                              backgroundColor: "black",
                              color: "white",
                              textTransform: "none",
                              "&:hover": {
                                backgroundColor: "gray",
                              },
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
                    </TableRow>
                  ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>

      {/* 成功・失敗メッセージを表示する汎用ダイアログ */}
      <Dialog open={dialogOpen} onClose={handleCloseDialog}>
        <DialogTitle>{"File Upload Status"}</DialogTitle>
        <DialogContent>
          <DialogContentText>{dialogMessage}</DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Mark as Complete 確認ダイアログ */}
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

      {/* プレビュー用ダイアログ */}
      <Dialog open={previewDialogOpen} onClose={handleClosePreviewDialog}>
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
