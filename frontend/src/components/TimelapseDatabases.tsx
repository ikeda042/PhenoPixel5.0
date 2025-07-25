import React, { useEffect, useState } from "react";
import {
  Container,
  Box,
  Typography,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Paper,
  Button,
  IconButton,
  Tooltip,
  Breadcrumbs,
  Link,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Backdrop,
  CircularProgress,
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import { useNavigate } from "react-router-dom";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";

interface ListDBResponse {
  databases: string[];
}

interface FieldsResponse {
  fields: string[];
}

const url_prefix = settings.url_prefix;

const TimelapseDatabases: React.FC = () => {
  const [databases, setDatabases] = useState<string[]>([]);
  /**
   * key: データベース名, value: そのDBのフィールド一覧
   * 例) { "some_db.db": ["Field_1", "Field_2", ...], ... }
   */
  const [dbFields, setDbFields] = useState<Record<string, string[]>>({});
  /**
   * key: データベース名, value: 選択されたフィールド
   * 例) { "some_db.db": "Field_1", ... }
   */
  const [selectedFields, setSelectedFields] = useState<Record<string, string>>(
    {}
  );

  /**
   * key: データベース名, value: 選択されたチャンネル
   * 例) { "some_db.db": "ph", ... }
   */
  const [selectedChannels, setSelectedChannels] = useState<
    Record<string, string>
  >({});

  /**
   * モーダル関連のステート
   */
  const [previewModalOpen, setPreviewModalOpen] = useState<boolean>(false);
  const [previewModalDbName, setPreviewModalDbName] = useState<string>("");
  const [previewModalUrl, setPreviewModalUrl] = useState<string>("");

  /**
   * ローディングスピナーの開閉フラグ
   */
  const [loading, setLoading] = useState<boolean>(false);

  const navigate = useNavigate();

  /**
   * コンポーネント描画時にデータベース一覧を取得
   */
  useEffect(() => {
    const fetchDatabases = async () => {
      try {
        setLoading(true);
        const response = await axios.get<ListDBResponse>(
          `${url_prefix}/tlengine/databases`
        );
        const dbList = response.data.databases;
        setDatabases(dbList);

        // 各DBのフィールド一覧を取得
        const fieldsObj: Record<string, string[]> = {};
        for (const dbName of dbList) {
          try {
            const fieldsRes = await axios.get<FieldsResponse>(
              `${url_prefix}/tlengine/databases/${dbName}/fields`
            );
            fieldsObj[dbName] = fieldsRes.data.fields;
          } catch (fieldsErr) {
            console.error("Failed to fetch fields for", dbName, fieldsErr);
            fieldsObj[dbName] = []; // 取得失敗時は空配列
          }
        }
        setDbFields(fieldsObj);
      } catch (error) {
        console.error("Failed to fetch databases:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchDatabases();
  }, []);

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
   * Access Database ボタン押下で遷移
   */
  const handleNavigate = (dbName: string) => {
    navigate(`/tlengine/databases?db_name=${dbName}`);
  };

  /**
   * Previewボタン押下時に呼ばれる想定の関数
   * 指定したフィールド & チャンネルのGIFを作成APIを呼び、モーダルでプレビュー表示する
   */
  const handlePreview = async (dbName: string) => {
    const field = selectedFields[dbName];
    const channel = selectedChannels[dbName] || "ph";

    if (!field) {
      alert(`No field selected for ${dbName}`);
      return;
    }

    // DB名 から ND2ファイル名を推定 (例: "foo_cells.db" -> "foo.nd2")
    const fileName = dbName.replace("_cells.db", ".nd2");

    try {
      setLoading(true);
      // GIFを取得 (バイナリ)
      const response = await axios.get(
        `${url_prefix}/tlengine/nd2_files/${fileName}/cells/${field}/gif?channel=${channel}`,
        {
          responseType: "arraybuffer",
        }
      );
      // Blob を生成
      const blob = new Blob([response.data], { type: "image/gif" });
      const blobUrl = URL.createObjectURL(blob);

      // モーダルを開いてプレビュー表示
      setPreviewModalDbName(dbName);
      setPreviewModalUrl(blobUrl);
      setPreviewModalOpen(true);
    } catch (err) {
      console.error("Failed to fetch GIF:", err);
      alert(`Failed to fetch GIF for ${dbName}, Field: ${field}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * フィールドがユーザにより切り替えられたとき
   */
  const handleFieldChange = (dbName: string, newField: string) => {
    setSelectedFields((prev) => ({ ...prev, [dbName]: newField }));
  };

  /**
   * チャンネルがユーザにより切り替えられたとき
   */
  const handleChannelChange = (dbName: string, newChannel: string) => {
    setSelectedChannels((prev) => ({ ...prev, [dbName]: newChannel }));
  };

  /**
   * モーダルを閉じる
   */
  const handleClosePreview = () => {
    setPreviewModalOpen(false);
    setPreviewModalDbName("");
    setPreviewModalUrl("");
  };

  return (
    <Container maxWidth={false} disableGutters>
      {/* ローディングスピナー */}
      <Backdrop
        sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={loading}
      >
        <CircularProgress color="inherit" />
      </Backdrop>

      <Box mt={2}>
        <Box mb={2}>
          <Breadcrumbs aria-label="breadcrumb">
            <Link underline="hover" color="inherit" href="/">
              Top
            </Link>
            <Link underline="hover" color="inherit" href="/tlengine/dbconsole">
              Database Console
            </Link>
          </Breadcrumbs>
        </Box>

        {/* データベース一覧テーブル */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell align="center"><b>Database Name</b></TableCell>
                <TableCell align="center"><b>Copy</b></TableCell>
                <TableCell align="center"><b>Preview</b></TableCell>
                <TableCell align="center"><b>Access Database</b></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {databases.map((database) => (
                <TableRow key={database}>
                  {/* Database Name */}
                  <TableCell align="center">{database}</TableCell>

                  {/* Copy to Clipboard */}
                  <TableCell align="center">
                    <Tooltip title="Copy to clipboard">
                      <IconButton
                        onClick={() => handleCopyToClipboard(database)}
                        sx={{ color: 'text.primary' }}
                      >
                        <ContentCopyIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>

                  {/* Preview */}
                  <TableCell align="center">
                    {/* フィールドの選択ドロップダウン */}
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel id={`select-field-label-${database}`}>
                        Field
                      </InputLabel>
                      <Select
                        labelId={`select-field-label-${database}`}
                        label="Field"
                        value={selectedFields[database] || ""}
                        onChange={(e) => handleFieldChange(database, e.target.value)}
                      >
                        {/* 「all」オプションを先頭に追加 */}
                        <MenuItem value="all">All</MenuItem>

                        {(dbFields[database] || []).length === 0 && (
                          <MenuItem value="" disabled>
                            No fields
                          </MenuItem>
                        )}
                        {(dbFields[database] || []).map((field) => (
                          <MenuItem key={field} value={field}>
                            {field}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    {/* チャンネルの選択ドロップダウン */}
                    <FormControl
                      size="small"
                      sx={{ minWidth: 80, ml: 1 }}
                    >
                      <InputLabel id={`select-channel-label-${database}`}>
                        Ch
                      </InputLabel>
                      <Select
                        labelId={`select-channel-label-${database}`}
                        label="Ch"
                        value={selectedChannels[database] || "ph"}
                        onChange={(e) =>
                          handleChannelChange(database, e.target.value)
                        }
                      >
                        <MenuItem value="ph">ph</MenuItem>
                        <MenuItem value="fluo1">fluo1</MenuItem>
                        <MenuItem value="fluo2">fluo2</MenuItem>
                      </Select>
                    </FormControl>

                    {/* プレビューボタン */}
                    <Button
                      variant="contained"
                      size="small"
                      sx={{
                        ml: 2,
                        backgroundColor: 'primary.main',
                        color: 'primary.contrastText',
                        minHeight: "36px",
                        fontSize: "0.8rem",
                        '&:hover': {
                          backgroundColor: 'primary.dark',
                        },
                      }}
                      onClick={() => handlePreview(database)}
                    >
                      Preview
                    </Button>
                  </TableCell>

                  {/* Access database */}
                  <TableCell align="center">
                    <IconButton
                      onClick={() => handleNavigate(database)}
                      sx={{ color: 'text.primary' }}
                    >
                      <Typography
                        sx={{
                          color: 'text.primary',
                          fontSize: "0.8rem",
                        }}
                      >
                        Access database
                      </Typography>
                      <NavigateNextIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>

      {/* プレビュー用モーダル */}
      <Dialog
        open={previewModalOpen}
        onClose={handleClosePreview}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Preview - {previewModalDbName}</DialogTitle>
        <DialogContent>
          {previewModalUrl && (
            <img
              src={previewModalUrl}
              alt="Preview GIF"
              style={{ width: "100%", maxHeight: "80vh", objectFit: "contain" }}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={handleClosePreview}
            variant="contained"
            sx={{
              backgroundColor: 'primary.main',
              color: 'primary.contrastText',
              '&:hover': {
                backgroundColor: 'primary.dark',
              },
            }}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default TimelapseDatabases;
