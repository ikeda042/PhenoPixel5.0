import React, { useEffect, useState } from "react";
import {
  Container,
  Box,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Grid,
  Card,
  CardHeader,
  CardMedia,
    Breadcrumbs,
    Link,
} from "@mui/material";
import axios from "axios";
import { useSearchParams } from "react-router-dom";
import { settings } from "../settings";

/**
 * /databases/{db_name}/fields のレスポンス
 */
interface GetFieldsResponse {
  fields: string[];
}

/**
 * /databases/{db_name}/fields/{field}/cell_numbers のレスポンス
 */
interface GetCellNumbersResponse {
  cell_numbers: number[];
}

const url_prefix = settings.url_prefix;

/**
 * タイムラプスGIFを表示し、
 * DB名、Field、CellNumberなどを選択/操作できるコンポーネント
 * 画像は ph, fluo1, fluo2 の3チャネルを同時に表示する。
 */
const TimelapseViewer: React.FC = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name"); // "?db_name=xxx" を取得

  // フィールド一覧・選択中のフィールド
  const [fields, setFields] = useState<string[]>([]);
  const [selectedField, setSelectedField] = useState<string>("");

  // セル番号一覧・選択中のセル番号
  const [cellNumbers, setCellNumbers] = useState<number[]>([]);
  const [selectedCellNumber, setSelectedCellNumber] = useState<number>(0);

  // 常に表示したいチャネル
  const channels = ["ph", "fluo1", "fluo1"] as const;

  // DB名が取れない場合のエラーハンドリング
  useEffect(() => {
    if (!dbName) {
      console.error("No db_name is specified in query parameters.");
      // 必要であればエラーメッセージ表示やリダイレクトなどを行う
    }
  }, [dbName]);

  /**
   * DBのフィールド一覧を取得
   */
  const fetchFields = async (dbName: string) => {
    try {
      const response = await axios.get<GetFieldsResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/fields`
      );
      setFields(response.data.fields);

      // フィールド一覧取得後、先頭要素をデフォルト選択
      if (response.data.fields.length > 0) {
        setSelectedField(response.data.fields[0]);
      }
    } catch (error) {
      console.error("Failed to fetch fields:", error);
    }
  };

  /**
   * 指定フィールドのセル番号一覧を取得
   */
  const fetchCellNumbers = async (dbName: string, field: string) => {
    try {
      const response = await axios.get<GetCellNumbersResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/fields/${field}/cell_numbers`
      );
      setCellNumbers(response.data.cell_numbers);

      // セル番号一覧取得後、先頭要素をデフォルト選択
      if (response.data.cell_numbers.length > 0) {
        setSelectedCellNumber(response.data.cell_numbers[0]);
      }
    } catch (error) {
      console.error("Failed to fetch cell numbers:", error);
    }
  };

  /**
   * コンポーネント初回表示時にフィールド一覧を取得
   */
  useEffect(() => {
    if (dbName) {
      fetchFields(dbName);
    }
  }, [dbName]);

  /**
   * フィールドが変わったらセル番号を取得
   */
  useEffect(() => {
    if (dbName && selectedField) {
      fetchCellNumbers(dbName, selectedField);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedField]);

  /**
   * セル番号を前後に移動する (UI 上の Prev / Next ボタン用)
   */
  const handlePrevCell = () => {
    if (cellNumbers.length === 0) return;
    const currentIndex = cellNumbers.indexOf(selectedCellNumber);
    if (currentIndex > 0) {
      setSelectedCellNumber(cellNumbers[currentIndex - 1]);
    }
  };

  const handleNextCell = () => {
    if (cellNumbers.length === 0) return;
    const currentIndex = cellNumbers.indexOf(selectedCellNumber);
    if (currentIndex >= 0 && currentIndex < cellNumbers.length - 1) {
      setSelectedCellNumber(cellNumbers[currentIndex + 1]);
    }
  };

  /**
   * チャネルごとにタイムラプスGIFの URL を組み立て
   */
  const gifUrls = channels.map((ch) =>
    dbName
      ? `${url_prefix}/tlengine/databases/${dbName}/cells/gif/${selectedField}/${selectedCellNumber}?channel=${ch}`
      : ""
  );

  return (
    <Container sx={{ py: 4 }}>
        <Box mb={2}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Link underline="hover" color="inherit" href="/tlengine/dbconsole">
            Database Console
          </Link>
          <Typography color="text.primary">{dbName}</Typography>
        </Breadcrumbs>
      </Box>
      <Box mb={2}>
        <Typography variant="h4" gutterBottom>
          Timelapse Viewer
        </Typography>
      </Box>

      {/* フィールド選択 */}
      <FormControl sx={{ mr: 2, minWidth: 120, mb: 2 }}>
        <InputLabel id="field-select-label">Field</InputLabel>
        <Select
          labelId="field-select-label"
          value={selectedField}
          label="Field"
          onChange={(e) => setSelectedField(e.target.value as string)}
        >
          {fields.map((field) => (
            <MenuItem key={field} value={field}>
              {field}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* セル番号選択 */}
      <FormControl sx={{ mr: 2, minWidth: 120, mb: 2 }}>
        <InputLabel id="cellnumber-select-label">Cell #</InputLabel>
        <Select
          labelId="cellnumber-select-label"
          value={selectedCellNumber}
          label="Cell #"
          onChange={(e) => setSelectedCellNumber(e.target.value as number)}
        >
          {cellNumbers.map((num) => (
            <MenuItem key={num} value={num}>
              {num}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Prev/Next ボタン */}
      <Box mb={4}>
        <Button variant="outlined" sx={{ mr: 2 }} onClick={handlePrevCell}>
          Prev Cell
        </Button>
        <Button variant="outlined" onClick={handleNextCell}>
          Next Cell
        </Button>
      </Box>

      {/* タイムラプスGIFの表示 */}
      {dbName ? (
        <Grid container spacing={3}>
          {gifUrls.map((url, idx) => (
            <Grid item xs={12} md={4} key={channels[idx]}>
              <Card>
                <CardHeader
                  title={`Channel: ${channels[idx]}`}
                  sx={{ pb: 0 }}
                />
                <CardMedia
                  component="img"
                  image={url}
                  alt={`timelapse-${channels[idx]}`}
                  sx={{ mt: 1 }}
                />
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Typography variant="body1">
          データがありません。DB名やフィールドが正しく指定されているか確認してください。
        </Typography>
      )}
    </Container>
  );
};

export default TimelapseViewer;
