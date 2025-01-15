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
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import { useNavigate } from "react-router-dom";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";

/**
 * /tlengine/databases のレスポンス用インターフェイス
 */
interface ListDBResponse {
  databases: string[];
}

const url_prefix = settings.url_prefix;

const TimelapseDatabases: React.FC = () => {
  const [databases, setDatabases] = useState<string[]>([]);
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
        setDatabases(response.data.databases);
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
   * 指定のDBに画面遷移 (クエリパラメータ付き)
   * Databases.tsx での handleNavigate を参考に
   */
  const handleNavigate = (dbName: string) => {
    // 例: /tlengine/databases?db_name=XXX
    navigate(`/tlengine/databases?db_name=${dbName}`);
  };

  return (
    <Container>
      <Box mt={2}>
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
          {/* もし従来の「Access Databases」ボタンが必要なら残す */}
          {/* 
          <Button variant="contained" onClick={() => navigate("/tlengine/databases")}>
            Access Databases
          </Button>
          */}
        </Box>

        {/* データベース一覧テーブル */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Database Name</TableCell>
                <TableCell>Copy</TableCell>
                <TableCell>Access Database</TableCell>
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
