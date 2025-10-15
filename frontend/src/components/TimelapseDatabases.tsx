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

type CavDrawMode =
  | "basic"
  | "pixel_sd"
  | "pixel_cv"
  | "contour_areas"
  | "area_vs_sd"
  | "area_vs_cv";

const cavDrawModeItems = [
  { value: "basic", label: "Basic" },
  { value: "pixel_sd", label: "Pixel SD" },
  { value: "pixel_cv", label: "Pixel CV" },
  { value: "contour_areas", label: "Contour Area" },
  { value: "area_vs_sd", label: "Area vs SD" },
  { value: "area_vs_cv", label: "Area vs CV" },
];

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
   * key: データベース名, value: CAV CSV 用に選択された draw mode
   */
  const [selectedDrawModes, setSelectedDrawModes] = useState<
    Record<string, CavDrawMode>
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
    if (window.isSecureContext && navigator.clipboard?.writeText) {
      navigator.clipboard
        .writeText(dbName)
        .then(() => {
          alert(`${dbName} copied to clipboard!`);
        })
        .catch((err) => {
          console.error("Failed to copy text: ", err);
        });
      return;
    }

    try {
      const textarea = document.createElement("textarea");
      textarea.value = dbName;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      const succeeded = document.execCommand("copy");
      document.body.removeChild(textarea);
      if (succeeded) {
        alert(`${dbName} copied to clipboard!`);
      } else {
        throw new Error("execCommand copy failed");
      }
    } catch (err) {
      console.error("Failed to copy text: ", err);
      alert("Failed to copy. Please copy manually.");
    }
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
   * CAV CSV の draw mode がユーザにより切り替えられたとき
   */
  const handleDrawModeChange = (dbName: string, newMode: CavDrawMode) => {
    setSelectedDrawModes((prev) => ({ ...prev, [dbName]: newMode }));
  };

  /**
   * 指定したDBのdead/aliveセルのCSVを一括ダウンロード
   */
  const handleBulkDownload = async (
    dbName: string,
    type: "dead" | "alive"
  ) => {
    try {
      setLoading(true);
      const isDead = type === "dead" ? 1 : 0;
      const drawMode = selectedDrawModes[dbName] || "basic";
      const channel = selectedChannels[dbName] || "ph";
      const response = await axios.get(
        `${url_prefix}/tlengine/databases/${dbName}/cells/csv?is_dead=${isDead}&draw_mode=${drawMode}&channel=${channel}&manual_label=1`,
        { responseType: "blob" }
      );
      const blob = new Blob([response.data], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${dbName}_${type}_${drawMode}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Failed to download CSV:", err);
      alert("Failed to download CSV");
    } finally {
      setLoading(false);
    }
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
                <TableCell align="center"><b>Bulk CSV</b></TableCell>
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

                  {/* CAV CSV */}
                  <TableCell align="center">
                    <FormControl size="small" sx={{ minWidth: 120, mr: 1 }}>
                      <InputLabel id={`select-drawmode-label-${database}`}>
                        Mode
                      </InputLabel>
                      <Select
                        labelId={`select-drawmode-label-${database}`}
                        label="Mode"
                        value={selectedDrawModes[database] || "basic"}
                        onChange={(e) =>
                          handleDrawModeChange(
                            database,
                            e.target.value as CavDrawMode
                          )
                        }
                      >
                        {cavDrawModeItems.map((item) => (
                          <MenuItem key={item.value} value={item.value}>
                            {item.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <FormControl
                      size="small"
                      sx={{ minWidth: 80, mr: 1 }}
                    >
                      <InputLabel id={`select-csv-channel-label-${database}`}>
                        Ch
                      </InputLabel>
                      <Select
                        labelId={`select-csv-channel-label-${database}`}
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
                    <Button
                      variant="contained"
                      size="small"
                      sx={{
                        backgroundColor: 'error.main',
                        color: 'error.contrastText',
                        fontSize: '0.8rem',
                        '&:hover': { backgroundColor: 'error.dark' },
                      }}
                      onClick={() => handleBulkDownload(database, 'dead')}
                    >
                      Dead CSV
                    </Button>
                    <Button
                      variant="contained"
                      size="small"
                      sx={{
                        ml: 1,
                        backgroundColor: 'success.main',
                        color: 'success.contrastText',
                        fontSize: '0.8rem',
                        '&:hover': { backgroundColor: 'success.dark' },
                      }}
                      onClick={() => handleBulkDownload(database, 'alive')}
                    >
                      Alive CSV
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
