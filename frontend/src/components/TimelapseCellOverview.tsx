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
 * DB名、Field、CellNumber、Channelなどを選択/操作できるコンポーネント
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

  // チャネル選択用 (必要に応じて増やす)
  const channels = ["ph", "fluo1", "fluo2"] as const;
  type ChannelType = typeof channels[number];
  const [selectedChannel, setSelectedChannel] = useState<ChannelType>("ph");

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

      // フィールド一覧取得後、先頭要素をデフォルト選択に
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

      // セル番号一覧取得後、先頭要素をデフォルト選択に
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
   * タイムラプスGIF表示用の URL を組み立て
   * - 例: /databases/{db_name}/cells/gif/{field}/{cell_number}?channel=ph
   */
  const gifUrl = dbName
    ? `${url_prefix}/tlengine/databases/${dbName}/cells/gif/${selectedField}/${selectedCellNumber}?channel=${selectedChannel}`
    : "";

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

  return (
    <Container>
      <Box mt={2}>
        <Typography variant="h5" gutterBottom>
          Timelapse Viewer
        </Typography>

        {/* フィールド選択 */}
        <FormControl sx={{ mr: 2, minWidth: 120 }}>
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

        {/* チャネル選択 */}
        <FormControl sx={{ mr: 2, minWidth: 120 }}>
          <InputLabel id="channel-select-label">Channel</InputLabel>
          <Select
            labelId="channel-select-label"
            value={selectedChannel}
            label="Channel"
            onChange={(e) => setSelectedChannel(e.target.value as ChannelType)}
          >
            {channels.map((ch) => (
              <MenuItem key={ch} value={ch}>
                {ch}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* セル番号選択 (ドロップダウンでもよいし、Prev/Next ボタンでも操作可能) */}
        <FormControl sx={{ mr: 2, minWidth: 120 }}>
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
        <Box mt={2}>
          <Button variant="outlined" sx={{ mr: 2 }} onClick={handlePrevCell}>
            Prev Cell
          </Button>
          <Button variant="outlined" onClick={handleNextCell}>
            Next Cell
          </Button>
        </Box>

        {/* タイムラプスGIFの表示 */}
        <Box mt={4}>
          {gifUrl ? (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                {`Field: ${selectedField}, Cell #: ${selectedCellNumber}, Channel: ${selectedChannel}`}
              </Typography>
              {/* GIF 表示 */}
              <img src={gifUrl} alt="timelapse" />
            </Box>
          ) : (
            <Typography variant="body1">
              データがありません。DB名やフィールドが正しく指定されているか確認してください。
            </Typography>
          )}
        </Box>
      </Box>
    </Container>
  );
};

export default TimelapseViewer;
