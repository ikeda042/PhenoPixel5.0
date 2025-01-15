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
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import { useNavigate } from "react-router-dom";

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
   * 「Access Databases」 ボタンクリック時
   * React Router を利用して /tlengine/databases に遷移
   */
  const handleAccessDatabases = () => {
    navigate("/tlengine/databases");
  };

  return (
    <Container>
      <Box mt={2}>
        {/* タイトル + ボタン */}
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h5" gutterBottom>
            Timelapse Databases
          </Typography>
          <Button variant="contained" onClick={handleAccessDatabases}>
            Access Databases
          </Button>
        </Box>

        {/* データベース一覧テーブル */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Database Name</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {databases.map((database) => (
                <TableRow key={database}>
                  <TableCell>{database}</TableCell>
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
