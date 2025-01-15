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
  const [selectedFields, setSelectedFields] = useState<Record<string, string>>({});

  /**
   * key: データベース名, value: 生成したGIFのURL (Blob URL)
   */
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});

  const navigate = useNavigate();

  /**
   * コンポーネント描画時にデータベース一覧を取得
   */
  useEffect(() => {
    const fetchDatabases = async () => {
      try {
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
   * 指定したフィールドのGIFを作成APIを呼び、Blob URLを作ってプレビューする
   */
  const handlePreview = async (dbName: string) => {
    const field = selectedFields[dbName];
    if (!field) {
      alert(`No field selected for ${dbName}`);
      return;
    }

    // DB名 から ND2ファイル名を逆引き (例: "foo_cells.db" -> "foo.nd2")
    const fileName = dbName.replace("_cells.db", ".nd2");

    try {
      // GIFを取得
      const response = await axios.get(
        `${url_prefix}/tlengine/nd2_files/${fileName}/cells/${field}/gif`,
        {
          responseType: "arraybuffer", // バイナリを取得する
        }
      );

      // Blob を生成
      const blob = new Blob([response.data], { type: "image/gif" });
      // ブラウザ上で表示できるURLに変換
      const blobUrl = URL.createObjectURL(blob);

      // previewUrls に格納
      setPreviewUrls((prev) => ({ ...prev, [dbName]: blobUrl }));
    } catch (err) {
      console.error("Failed to fetch GIF:", err);
      alert(`Failed to fetch GIF for ${dbName}, Field: ${field}`);
    }
  };

  /**
   * プレビューで使用するフィールドがユーザにより切り替えられたとき
   */
  const handleFieldChange = (dbName: string, newField: string) => {
    setSelectedFields((prev) => ({ ...prev, [dbName]: newField }));
  };

  return (
    <Container>
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
        {/* タイトル */}
        <Box
          display="flex"
          alignItems="center"
          justifyContent="space-between"
          mb={2}
        >
          <Typography variant="h5" gutterBottom>
            Timelapse Databases
          </Typography>
        </Box>

        {/* データベース一覧テーブル */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Database Name</TableCell>
                <TableCell>Copy</TableCell>
                <TableCell>Access Database</TableCell>
                {/* 新しく追加する preview カラム */}
                <TableCell>Preview</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {databases.map((database) => (
                <TableRow key={database}>
                  {/* Database Name */}
                  <TableCell>{database}</TableCell>

                  {/* Copy to Clipboard */}
                  <TableCell>
                    <Tooltip title="Copy to clipboard">
                      <IconButton onClick={() => handleCopyToClipboard(database)}>
                        <ContentCopyIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>

                  {/* Access database */}
                  <TableCell>
                    <IconButton onClick={() => handleNavigate(database)}>
                      <Typography>Access database</Typography>
                      <NavigateNextIcon />
                    </IconButton>
                  </TableCell>

                  {/* Preview */}
                  <TableCell>
                    {/* フィールドの選択ドロップダウン */}
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <InputLabel id={`select-label-${database}`}>Field</InputLabel>
                      <Select
                        labelId={`select-label-${database}`}
                        label="Field"
                        value={selectedFields[database] || ""}
                        onChange={(e) => handleFieldChange(database, e.target.value)}
                      >
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
                    {/* プレビューボタン */}
                    <Button
                      variant="contained"
                      size="small"
                      sx={{ ml: 2 }}
                      onClick={() => handlePreview(database)}
                    >
                      Preview
                    </Button>
                    {/* 取得したGIFのプレビュー表示 */}
                    {previewUrls[database] && (
                      <Box mt={1}>
                        <img
                          src={previewUrls[database]}
                          alt={`${database}-gif`}
                          style={{ maxHeight: "200px", maxWidth: "200px" }}
                        />
                      </Box>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Container>
  );
};

export default TimelapseDatabases;
