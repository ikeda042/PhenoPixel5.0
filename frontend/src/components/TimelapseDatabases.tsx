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
        // Promise.all で並行して取得しても良いし、逐次取得でもOK
        const fieldsObj: Record<string, string[]> = {};
        for (const dbName of dbList) {
          try {
            const fieldsRes = await axios.get<FieldsResponse>(
              `${url_prefix}/tlengine/databases/${dbName}/fields`
            );
            fieldsObj[dbName] = fieldsRes.data.fields;
          } catch (fieldsErr) {
            console.error("Failed to fetch fields for", dbName, fieldsErr);
            fieldsObj[dbName] = []; // フィールドが取得できなければ空配列にしておく
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
   * 実際には、ここでGIF作成APIを呼ぶ、あるいはプレビュー用のURLを開くなど自由に拡張してください
   */
  const handlePreview = (dbName: string) => {
    const field = selectedFields[dbName];
    if (!field) {
      alert(`No field selected for ${dbName}`);
      return;
    }
    // ここで自由にプレビュー処理を実装
    alert(`Preview for DB: ${dbName}, Field: ${field}`);
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
          {/* 必要に応じてボタン等を追加 */}
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
