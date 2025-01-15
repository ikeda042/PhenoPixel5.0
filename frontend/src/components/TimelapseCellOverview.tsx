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
  Card,
  CardHeader,
  CardContent,
  CardMedia,
  Breadcrumbs,
  Link,
  useTheme,
  useMediaQuery,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Checkbox,
  FormControlLabel,
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

/**
 * セルを取得するエンドポイント (/databases/{db_name}/cells/by_field/{field}/cell_number/{cell_number}) のレスポンス
 */
interface CellData {
  id: number;
  cell_id: string;
  field: string;
  time: number;
  cell: number;
  area: number;
  perimeter: number;
  manual_label?: number; 
  is_dead?: boolean;
}

interface GetCellsResponse {
  cells: CellData[];
}

const url_prefix = settings.url_prefix;

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

  // 今表示中のセル情報
  const [currentCellData, setCurrentCellData] = useState<CellData | null>(null);

  // manual_label のセレクトボックス用 (N/A, 1, 2, 3, 4)
  const manualLabelOptions = ["N/A", "1", "2", "3", "4"];

  // GIF の再生タイミングを揃えるためのキー
  const [reloadKey, setReloadKey] = useState<number>(0);

  // 「All Cells」プレビュー用のモーダル管理
  const [openModal, setOpenModal] = useState<boolean>(false);
  const [loadingAllCells, setLoadingAllCells] = useState<boolean>(false);
  const [allCellsGifUrl, setAllCellsGifUrl] = useState<string>("");

  // 表示したいチャネル（ph, fluo1, fluo2）
  const channels = ["ph", "fluo1", "fluo2"] as const;

  // DB名が取れない場合のエラーハンドリング
  useEffect(() => {
    if (!dbName) {
      console.error("No db_name is specified in query parameters.");
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
   * 現在選択中の Field & Cell Number で、細胞情報を取得
   */
  const fetchCurrentCellData = async () => {
    if (!dbName || !selectedField || !selectedCellNumber) {
      setCurrentCellData(null);
      return;
    }

    try {
      // by_field + cell_number のエンドポイントを呼び、CellData を取得
      const response = await axios.get<GetCellsResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/cells/by_field/${selectedField}/cell_number/${selectedCellNumber}`
      );

      const cells = response.data.cells;
      if (cells.length > 0) {
        // 先頭のセル情報を格納 (time が複数ある場合、例として1つだけ表示)
        setCurrentCellData(cells[0]);
      } else {
        setCurrentCellData(null);
      }
    } catch (error) {
      console.error("Failed to fetch current cell data:", error);
      setCurrentCellData(null);
    }
  };

  /**
   * manual_label を変更したら自動的にPATCH
   */
  const handleChangeManualLabel = async (value: string) => {
    if (!dbName || !currentCellData) return;

    // "N/A" はサーバー的には何もない値として扱いたい想定であれば、例えば label="" を送るなど
    // ここでは "N/A" を文字列としてそのまま送っている例です
    const patchLabel = value === "N/A" ? "N/A" : value; 

    try {
      const baseCellId = currentCellData.cell_id;
      await axios.patch(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${baseCellId}/label?label=${patchLabel}`
      );
      // 成功後、最新データを再取得
      fetchCurrentCellData();
    } catch (error) {
      console.error("Failed to update manual_label:", error);
    }
  };

  /**
   * is_dead のチェックが変わったら自動的にPATCH
   */
  const handleChangeIsDead = async (checked: boolean) => {
    if (!dbName || !currentCellData) return;

    try {
      const baseCellId = currentCellData.cell_id;
      await axios.patch(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${baseCellId}/dead?is_dead=${checked}`
      );
      // 成功後、最新データを再取得
      fetchCurrentCellData();
    } catch (error) {
      console.error("Failed to update is_dead:", error);
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
   * フィールド or セル番号が変わったら、細胞情報を取得
   */
  useEffect(() => {
    fetchCurrentCellData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dbName, selectedField, selectedCellNumber]);

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

  /**
   * 「Field すべての細胞の GIF プレビュー」を取得するボタン (モーダル + ローディング)
   */
  const handlePreviewAllCells = async () => {
    if (!dbName || !selectedField) {
      console.error("DB名やFieldが未選択です。");
      return;
    }

    // dbName から nd2 ファイル名を導出 (例: sample_cells.db -> sample.nd2)
    const fileName = dbName.replace("_cells.db", "") + ".nd2";

    // モーダルを開き & ローディング開始
    setOpenModal(true);
    setLoadingAllCells(true);
    setAllCellsGifUrl("");

    try {
      // バイナリ(画像)として取得
      const response = await axios.get(
        `${url_prefix}/tlengine/nd2_files/${fileName}/cells/${selectedField}/gif`,
        { responseType: "blob" }
      );

      // Blob を生成してプレビュー用のURLを作成
      const blobUrl = URL.createObjectURL(response.data);
      setAllCellsGifUrl(blobUrl);
    } catch (error) {
      console.error("Failed to fetch all-cells gif:", error);
    } finally {
      setLoadingAllCells(false);
    }
  };

  return (
    <>
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

            {/* 全細胞の GIF を取得するボタン */}
            <Button
              variant="contained"
              sx={{
                backgroundColor: "#444",
                color: "#fff",
                "&:hover": {
                  backgroundColor: "#666",
                },
              }}
              onClick={handlePreviewAllCells}
            >
              Preview All Cells
            </Button>
          </Box>
        </Box>

        {/* Current Cell Info */}
        {currentCellData && (
          <Box mb={3}>
            {/* manual_label の選択セレクトボックス：選んだら即 PATCH */}
            <Box mt={2}>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel id="manual-label-select-label">
                  manual_label
                </InputLabel>
                <Select
                  labelId="manual-label-select-label"
                  label="manual_label"
                  value={
                    currentCellData.manual_label !== undefined
                      ? String(currentCellData.manual_label)
                      : "N/A"
                  }
                  onChange={(e) => handleChangeManualLabel(e.target.value)}
                >
                  {manualLabelOptions.map((option) => (
                    <MenuItem key={option} value={option}>
                      {option}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>

            {/* is_dead のチェックボックス：操作したら即 PATCH */}
            <Box mt={2}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={currentCellData.is_dead ?? false}
                    onChange={(e) => handleChangeIsDead(e.target.checked)}
                  />
                }
                label="is_dead"
              />
            </Box>
          </Box>
        )}

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

      {/* すべての Cells GIF プレビュー用モーダル */}
      <Dialog
        open={openModal}
        onClose={() => setOpenModal(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>All Cells Preview</DialogTitle>
        <DialogContent>
          {loadingAllCells ? (
            <Box
              display="flex"
              justifyContent="center"
              alignItems="center"
              minHeight="200px"
            >
              <CircularProgress />
            </Box>
          ) : (
            <Box textAlign="center">
              {allCellsGifUrl ? (
                <img
                  src={allCellsGifUrl}
                  alt="All Cells GIF"
                  style={{ maxWidth: "100%", borderRadius: 4 }}
                />
              ) : (
                <Typography>No data available.</Typography>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenModal(false)} color="primary">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default TimelapseViewer;
