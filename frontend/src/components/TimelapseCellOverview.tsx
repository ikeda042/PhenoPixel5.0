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
  CardContent,
  CardMedia,
  Breadcrumbs,
  Link,
  useTheme,
  useMediaQuery,
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
 * 画像は ph, fluo1, fluo2 の3チャネルを隣接して表示する。
 */
const TimelapseViewer: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name"); // "?db_name=xxx" を取得

  // フィールド一覧・選択中のフィールド
  const [fields, setFields] = useState<string[]>([]);
  const [selectedField, setSelectedField] = useState<string>("");

  // セル番号一覧・選択中のセル番号
  const [cellNumbers, setCellNumbers] = useState<number[]>([]);
  const [selectedCellNumber, setSelectedCellNumber] = useState<number>(0);

  // GIF の再生タイミングを揃えるためのキー
  const [reloadKey, setReloadKey] = useState<number>(0);

  // 表示したいチャネル（ph, fluo1, fluo2）
  const channels = ["ph", "fluo1", "fluo2"] as const;

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
   * いずれかが変わったら GIF を再同期する
   */
  useEffect(() => {
    // フィールドやセル番号が変更されたタイミングで、
    // GIF をリロードして再生タイミングを揃える
    setReloadKey((prev) => prev + 1);
  }, [dbName, selectedField, selectedCellNumber]);

  /**
   * チャネルごとにタイムラプスGIFの URL を組み立て
   */
  const gifUrls = channels.map((ch) =>
    dbName
      ? `${url_prefix}/tlengine/databases/${dbName}/cells/gif/${selectedField}/${selectedCellNumber}?channel=${ch}`
      : ""
  );

  return (
    <Container
      sx={{
        py: 4,
        backgroundColor: "#f9f9f9",
        minHeight: "100vh",
      }}
      maxWidth="xl"
    >
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

      <Box mb={3}>
        <Typography variant="h4" gutterBottom fontWeight="bold">
          Timelapse Viewer
        </Typography>
      </Box>

      {/* フィールド＆セル番号選択 */}
      <Box
        display="flex"
        flexWrap="wrap"
        alignItems="center"
        gap={2}
        mb={3}
        flexDirection={isMobile ? "column" : "row"}
      >
        <FormControl sx={{ minWidth: 120 }}>
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

        <FormControl sx={{ minWidth: 120 }}>
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
        <Box display="flex" flexDirection="row" gap={2}>
          <Button
            variant="contained"
            sx={{
              backgroundColor: "#000",
              color: "#fff",
              "&:hover": {
                backgroundColor: "#333",
              },
            }}
            onClick={handlePrevCell}
          >
            Prev Cell
          </Button>
          <Button
            variant="contained"
            sx={{
              backgroundColor: "#000",
              color: "#fff",
              "&:hover": {
                backgroundColor: "#333",
              },
            }}
            onClick={handleNextCell}
          >
            Next Cell
          </Button>
        </Box>
      </Box>

      {/* タイムラプスGIFの表示（3チャネルをまとめて1つのブロックとして表示） */}
      {dbName ? (
        <Card
          sx={{
            borderRadius: 2,
            boxShadow: 3,
            backgroundColor: "#fff",
          }}
        >
          <CardHeader
            title="Channels: ph / fluo1 / fluo2"
            sx={{
              pb: 1,
              "& .MuiCardHeader-title": {
                fontWeight: "bold",
              },
            }}
          />
          <CardContent>
            <Box
              display="flex"
              flexDirection={isMobile ? "column" : "row"}
              gap={2}
              justifyContent="center"
              alignItems="center"
            >
              {gifUrls.map((url, idx) => (
                <CardMedia
                  key={`${channels[idx]}-${reloadKey}`}
                  component="img"
                  image={url}
                  alt={`timelapse-${channels[idx]}`}
                  sx={{
                    maxWidth: isMobile ? "100%" : "30%",
                    borderRadius: 2,
                    objectFit: "contain",
                  }}
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      ) : (
        <Typography variant="body1" mt={2}>
          データがありません。DB名やフィールドが正しく指定されているか確認してください。
        </Typography>
      )}
    </Container>
  );
};

export default TimelapseViewer;
