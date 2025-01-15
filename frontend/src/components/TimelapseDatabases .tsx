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
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

/**
 * /tlengine/databases のレスポンス用インターフェイス
 */
interface ListDBResponse {
  databases: string[];
}

const url_prefix = settings.url_prefix;

const TimelapseDatabases: React.FC = () => {
  const [databases, setDatabases] = useState<string[]>([]);

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

  return (
    <Container>
      <Box mt={2}>
        <Typography variant="h5" gutterBottom>
          Timelapse Databases
        </Typography>
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
