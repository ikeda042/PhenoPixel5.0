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
  Radio,
  RadioGroup,
  Grid,
} from "@mui/material";
import { ArrowBack, ArrowForward } from "@mui/icons-material";
import DownloadIcon from "@mui/icons-material/Download";
import axios from "axios";
import { useSearchParams } from "react-router-dom";
import { settings } from "../settings";

// Chart.js 関連
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  ChartData,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface GetFieldsResponse {
  fields: string[];
}

interface GetCellNumbersResponse {
  cell_numbers: number[];
}

interface CellDataByFieldNumber {
  id: number;
  cell_id: string;
  base_cell_id: string;
  field: string;
  time: number;
  cell: number;
  area: number;
  perimeter: number;
}

interface GetCellsResponseByFieldNumber {
  cells: CellDataByFieldNumber[];
}

interface CellDataById {
  id: number;
  cell_id: string;
  base_cell_id: string;
  field: string;
  time: number;
  cell: number;
  area: number;
  perimeter: number;
  manual_label?: string | number;
  is_dead?: number;
}

// 輪郭面積
interface ContourArea {
  frame: number;
  area: number;
}
interface GetContourAreasResponse {
  areas: number[];
}

// Pixel SD
interface PixelSD {
  frame: number;
  sd: number;
}
interface GetPixelSDResponse {
  sds: number[];
}

// Pixel CV
interface PixelCV {
  frame: number;
  cv: number;
}
interface GetPixelCVResponse {
  cvs: number[];
}

const url_prefix = settings.url_prefix;

const TimelapseViewer: React.FC = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name");

  // フィールド一覧・選択中のフィールド
  const [fields, setFields] = useState<string[]>([]);
  const [selectedField, setSelectedField] = useState<string>("");

  // セル番号一覧・選択中のセル番号
  const [cellNumbers, setCellNumbers] = useState<number[]>([]);
  const [selectedCellNumber, setSelectedCellNumber] = useState<number>(0);

  // 今表示中のセル詳細
  const [currentCellData, setCurrentCellData] = useState<CellDataById | null>(null);

  // manual_label のセレクトボックス用
  const manualLabelOptions = ["N/A", "1", "2", "3"];

  // 「全 GIF を同じタイミングで再生開始する」ためのキー
  const [reloadKey, setReloadKey] = useState<number>(0);

  // 「All Cells」プレビュー用モーダル
  const [openModal, setOpenModal] = useState<boolean>(false);
  const [loadingAllCells, setLoadingAllCells] = useState<boolean>(false);
  const [allCellsGifUrl, setAllCellsGifUrl] = useState<string>("");

  // 画像チャンネル (GIF 用)
  const channels = ["ph", "fluo1", "fluo2"] as const;

  // 輪郭面積グラフ用データ
  const [contourAreas, setContourAreas] = useState<ContourArea[]>([]);

  // Pixel SD data
  const [pixelSDs, setPixelSDs] = useState<PixelSD[]>([]);
  // Pixel CV data
  const [pixelCVs, setPixelCVs] = useState<PixelCV[]>([]);

  // ★ 描画モード: ContourAreas / PixelSD / PixelCV / Replot / TimecoursePNG
  type DrawMode = "ContourAreas" | "PixelSD" | "PixelCV" | "Replot" | "TimecoursePNG";
  const [drawMode, setDrawMode] = useState<DrawMode>("ContourAreas");

  // Replot 用: ph / fluo1 / fluo2
  const [replotChannel, setReplotChannel] = useState<"ph" | "fluo1" | "fluo2">("ph");

  // PixelSD 用 channel
  const [pixelSDChannel, setPixelSDChannel] = useState<"ph" | "fluo1" | "fluo2">("ph");

  // ★ TimecoursePNG 用チャネルモード
  type ChannelMode =
    | "ph"
    | "ph_replot"
    | "fluo1"
    | "fluo1_replot"
    | "fluo2"
    | "fluo2_replot"
    // ▼▼▼ ここが追加箇所 ▼▼▼
    | "all_channels";

  const [timecourseChannelMode, setTimecourseChannelMode] = useState<ChannelMode>("ph");

  // 画像ロード完了数
  const [imagesLoadedCount, setImagesLoadedCount] = useState(0);

  // ==================
  // 初期処理
  // ==================
  useEffect(() => {
    if (!dbName) {
      console.error("No db_name is specified in query parameters.");
    }
  }, [dbName]);

  const fetchFields = async (dbName: string) => {
    try {
      const response = await axios.get<GetFieldsResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/fields`
      );
      setFields(response.data.fields);
      if (response.data.fields.length > 0) {
        setSelectedField(response.data.fields[0]);
      }
    } catch (error) {
      console.error("Failed to fetch fields:", error);
    }
  };

  const fetchCellNumbers = async (dbName: string, field: string) => {
    try {
      const response = await axios.get<GetCellNumbersResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/fields/${field}/cell_numbers`
      );
      setCellNumbers(response.data.cell_numbers);
      if (response.data.cell_numbers.length > 0) {
        setSelectedCellNumber(response.data.cell_numbers[0]);
      }
    } catch (error) {
      console.error("Failed to fetch cell numbers:", error);
    }
  };

  const fetchCellDataById = async (cellId: string) => {
    if (!dbName) return null;
    try {
      const response = await axios.get<CellDataById>(
        `${url_prefix}/tlengine/databases/${dbName}/cells/by_id/${cellId}`
      );
      return response.data;
    } catch (error) {
      console.error("Failed to fetch cell data by cell_id:", error);
      return null;
    }
  };

  const fetchCurrentCellData = async () => {
    if (!dbName || !selectedField || !selectedCellNumber) {
      setCurrentCellData(null);
      return;
    }
    try {
      const response = await axios.get<GetCellsResponseByFieldNumber>(
        `${url_prefix}/tlengine/databases/${dbName}/cells/by_field/${selectedField}/cell_number/${selectedCellNumber}`
      );
      const cells = response.data.cells;
      if (cells.length === 0) {
        setCurrentCellData(null);
        return;
      }
      // cell_id を使って詳細情報を取得する
      const cellId = cells[0].cell_id;
      const detail = await fetchCellDataById(cellId);
      if (detail) {
        setCurrentCellData(detail);
      } else {
        setCurrentCellData(null);
      }
    } catch (error) {
      console.error("Failed to fetch current cell data:", error);
      setCurrentCellData(null);
    }
  };

  // manual_label 更新
  const handleChangeManualLabel = async (value: string) => {
    if (!dbName || !currentCellData) return;
    try {
      const baseCellId = currentCellData.base_cell_id;
      await axios.patch(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${baseCellId}/label?label=${value}`
      );
      // 更新後再取得
      await fetchCurrentCellData();
    } catch (error) {
      console.error("Failed to update manual_label:", error);
    }
  };

  // is_dead 更新
  const handleChangeIsDead = async (checked: boolean) => {
    if (!dbName || !currentCellData) return;
    try {
      const baseCellId = currentCellData.base_cell_id;
      const isDeadValue = checked ? 1 : 0;
      await axios.patch(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${baseCellId}/dead/${isDeadValue}`
      );
      await fetchCurrentCellData();
    } catch (error) {
      console.error("Failed to update is_dead:", error);
    }
  };

  // フィールド・セル番号一覧取得
  useEffect(() => {
    if (dbName) {
      fetchFields(dbName);
    }
  }, [dbName]);

  useEffect(() => {
    if (dbName && selectedField) {
      fetchCellNumbers(dbName, selectedField);
    }
  }, [dbName, selectedField]);

  useEffect(() => {
    fetchCurrentCellData();
  }, [dbName, selectedField, selectedCellNumber]);

  // PixelSD graph
  const fetchPixelSDs = async (controller: AbortController) => {
    if (!dbName || !selectedField || !selectedCellNumber) {
      setPixelSDs([]);
      return;
    }
    try {
      const response = await axios.get<GetPixelSDResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/pixel_sd?channel=${pixelSDChannel}`,
        { signal: controller.signal }
      );
      if (controller.signal.aborted) return;
      const converted = response.data.sds.map((value, index) => ({
        frame: index,
        sd: value,
      }));
      setPixelSDs(converted);
    } catch (error) {
      if (!controller.signal.aborted) {
        console.error("Failed to fetch pixel sd:", error);
        setPixelSDs([]);
      }
    }
  };

  // PixelCV graph
  const fetchPixelCVs = async (controller: AbortController) => {
    if (!dbName || !selectedField || !selectedCellNumber) {
      setPixelCVs([]);
      return;
    }
    try {
      const response = await axios.get<GetPixelCVResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/pixel_cv?channel=${pixelSDChannel}`,
        { signal: controller.signal }
      );
      if (controller.signal.aborted) return;
      const converted = response.data.cvs.map((value, index) => ({
        frame: index,
        cv: value,
      }));
      setPixelCVs(converted);
    } catch (error) {
      if (!controller.signal.aborted) {
        console.error("Failed to fetch pixel cv:", error);
        setPixelCVs([]);
      }
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    fetchPixelSDs(controller);
    return () => controller.abort();
  }, [dbName, selectedField, selectedCellNumber, pixelSDChannel]);

  useEffect(() => {
    const controller = new AbortController();
    fetchPixelCVs(controller);
    return () => controller.abort();
  }, [dbName, selectedField, selectedCellNumber, pixelSDChannel]);

  // Prev / Next
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
    if (currentIndex < cellNumbers.length - 1) {
      setSelectedCellNumber(cellNumbers[currentIndex + 1]);
    } else {
      // 最後のセル → 次のフィールドへ
      const fieldIndex = fields.indexOf(selectedField);
      if (fieldIndex < fields.length - 1) {
        setSelectedField(fields[fieldIndex + 1]);
      } else {
        console.log("すべてのフィールド・セルを見終わりました。");
      }
    }
  };

  // 画像ロードカウント
  useEffect(() => {
    setImagesLoadedCount(0);
    setReloadKey((prev) => prev + 1);
  }, [dbName, selectedField, selectedCellNumber, drawMode, replotChannel, timecourseChannelMode]);

  const handleImageLoad = () => {
    setImagesLoadedCount((prev) => prev + 1);
  };

  // 通常GIF 3枚
  const normalGifUrls = channels.map((ch) => {
    if (!dbName) return "";
    return `${url_prefix}/tlengine/databases/${dbName}/cells/gif/${selectedField}/${selectedCellNumber}?channel=${ch}&duration=200&_syncKey=${reloadKey}`;
  });

  // Replot 用 GIF
  const replotGifUrl = dbName
    ? `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/replot?channel=${replotChannel}&degree=4&duration=200&_syncKey=${reloadKey}`
    : "";

  // ★ TimecoursePNG 用 URL (channel_mode="all_channels" でエンドポイントを切り替え)
  const timecoursePngUrl = dbName
    ? timecourseChannelMode === "all_channels"
      ? `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/timecourse_png/all_channels?degree=0&draw_contour=true&_syncKey=${reloadKey}`
      : `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/timecourse_png?channel_mode=${timecourseChannelMode}&degree=0&draw_contour=true&_syncKey=${reloadKey}`
    : "";

  // ロード対象URL
  const allGifUrls = (() => {
    let baseUrls = [...normalGifUrls];
    if (drawMode === "Replot") {
      baseUrls.push(replotGifUrl);
    }
    if (drawMode === "TimecoursePNG") {
      baseUrls.push(timecoursePngUrl);
    }
    return baseUrls;
  })();

  const allLoaded = imagesLoadedCount === allGifUrls.length;

  // キーボード操作
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!currentCellData) return;
      switch (e.key) {
        case "d":
          e.preventDefault();
          handleChangeIsDead(currentCellData.is_dead !== 1);
          break;
        case "n":
          e.preventDefault();
          handleChangeManualLabel("N/A");
          break;
        case "1":
        case "2":
        case "3":
          e.preventDefault();
          handleChangeManualLabel(e.key);
          break;
        case "Enter":
          e.preventDefault();
          handleNextCell();
          break;
        case " ":
          e.preventDefault();
          handlePrevCell();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [currentCellData]);

  // All Cells Preview
  const handlePreviewAllCells = async () => {
    if (!dbName || !selectedField) return;
    const fileName = dbName.replace("_cells.db", "") + ".nd2";
    setOpenModal(true);
    setLoadingAllCells(true);
    setAllCellsGifUrl("");

    try {
      const response = await axios.get(
        `${url_prefix}/tlengine/nd2_files/${fileName}/cells/${selectedField}/gif`,
        { responseType: "blob" }
      );
      const blobUrl = URL.createObjectURL(response.data);
      setAllCellsGifUrl(blobUrl);
    } catch (error) {
      console.error("Failed to fetch all-cells gif:", error);
    } finally {
      setLoadingAllCells(false);
    }
  };

  const handleDownloadCsv = async () => {
    if (!dbName) return;
    let endpoint = "";
    if (drawMode === "PixelSD") {
      endpoint = `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/pixel_sd/csv?channel=${pixelSDChannel}`;
    } else if (drawMode === "PixelCV") {
      endpoint = `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/pixel_cv/csv?channel=${pixelSDChannel}`;
    } else if (drawMode === "ContourAreas") {
      endpoint = `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/contour_areas/csv`;
    } else {
      return;
    }
    try {
      const response = await axios.get(endpoint, { responseType: "blob" });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute(
        "download",
        `${dbName}_${selectedField}_${selectedCellNumber}_${drawMode}.csv`
      );
      document.body.appendChild(link);
      link.click();
      link?.parentNode?.removeChild(link);
    } catch (error) {
      console.error("Failed to download CSV:", error);
    }
  };

  // ContourAreas グラフ
  const fetchContourAreas = async (controller: AbortController) => {
    if (!dbName || !selectedField || !selectedCellNumber) {
      setContourAreas([]);
      return;
    }
    try {
      const response = await axios.get<GetContourAreasResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/contour_areas`,
        { signal: controller.signal }
      );
      if (controller.signal.aborted) return;
      const converted = response.data.areas.map((value, index) => ({
        frame: index,
        area: value,
      }));
      setContourAreas(converted);
    } catch (error) {
      if (!controller.signal.aborted) {
        console.error("Failed to fetch contour areas:", error);
        setContourAreas([]);
      }
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    fetchContourAreas(controller);
    return () => controller.abort();
  }, [dbName, selectedField, selectedCellNumber]);

  const contourAreasChartData: ChartData<"line"> = {
    labels: contourAreas.map((ca) => ca.frame),
    datasets: [
      {
        label: "Contour Area",
        data: contourAreas.map((ca) => ca.area),
        fill: false,
        borderColor: "rgba(75,192,192,1)",
        tension: 0.1,
      },
    ],
  };

  const contourAreasChartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: "Contour Areas (frame vs. area)",
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Frame (Index)",
        },
      },
      y: {
        title: {
          display: true,
          text: "Area",
        },
        min: 0,
      },
    },
  };

  const pixelSDChartData: ChartData<"line"> = {
    labels: pixelSDs.map((p) => p.frame),
    datasets: [
      {
        label: "Pixel SD",
        data: pixelSDs.map((p) => p.sd),
        fill: false,
        borderColor: "rgba(255,99,132,1)",
        tension: 0.1,
      },
    ],
  };

  const pixelCVChartData: ChartData<"line"> = {
    labels: pixelCVs.map((p) => p.frame),
    datasets: [
      {
        label: "Pixel CV",
        data: pixelCVs.map((p) => p.cv),
        fill: false,
        borderColor: "rgba(54,162,235,1)",
        tension: 0.1,
      },
    ],
  };

  const pixelSDChartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: "SD of cell pixels (frame vs. SD)",
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Frame (Index)",
        },
      },
      y: {
        title: {
          display: true,
          text: "SD",
        },
        min: 0,
      },
    },
  };

  const pixelCVChartOptions: ChartOptions<"line"> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: "CV of cell pixels (frame vs. CV)",
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Frame (Index)",
        },
      },
      y: {
        title: {
          display: true,
          text: "CV",
        },
        min: 0,
      },
    },
  };

  return (
    <>
      <Container sx={{ py: 4, backgroundColor: "#fff", minHeight: "100vh" }} maxWidth="xl">
        {/* パンくず */}
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

        {/* Field, Cell選択など */}
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

          {currentCellData && (
            <>
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel id="manual-label-select-label">manual_label</InputLabel>
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

              <FormControl>
                <RadioGroup
                  row
                  value={currentCellData.is_dead === 1 ? "dead" : "alive"}
                  onChange={(e) =>
                    handleChangeIsDead(e.target.value === "dead")
                  }
                >
                  <FormControlLabel
                    value="alive"
                    control={<Radio color="primary" />}
                    label="Alive"
                  />
                  <FormControlLabel
                    value="dead"
                    control={<Radio color="error" />}
                    label="Dead"
                  />
                </RadioGroup>
              </FormControl>

              <Typography variant="body2" sx={{ ml: 2 }}>
                BaseID: {currentCellData.base_cell_id}
              </Typography>
            </>
          )}

          {/* Prev / Next */}
          <Button
            variant="contained"
            startIcon={<ArrowBack />}
            sx={{ backgroundColor: "#000", color: "#fff", "&:hover": { backgroundColor: "#333" } }}
            onClick={handlePrevCell}
          >
            Prev
          </Button>
          <Button
            variant="contained"
            endIcon={<ArrowForward />}
            sx={{ backgroundColor: "#000", color: "#fff", "&:hover": { backgroundColor: "#333" } }}
            onClick={handleNextCell}
          >
            Next
          </Button>

          <Button
            variant="contained"
            sx={{ backgroundColor: "#444", color: "#fff", "&:hover": { backgroundColor: "#666" } }}
            onClick={handlePreviewAllCells}
          >
            Preview All Cells
          </Button>
        </Box>

        {/* 描画モード切り替え */}
        <Box mb={2} display="flex" alignItems="center" gap={2}>
          <FormControl sx={{ minWidth: 180 }}>
            <InputLabel id="draw-mode-select-label">DrawMode</InputLabel>
            <Select
              labelId="draw-mode-select-label"
              value={drawMode}
              label="描画モード"
              onChange={(e) => setDrawMode(e.target.value as DrawMode)}
            >
              <MenuItem value="ContourAreas">ContourAreas</MenuItem>
              <MenuItem value="PixelSD">PixelSD</MenuItem>
              <MenuItem value="PixelCV">PixelCV</MenuItem>
              <MenuItem value="Replot">Replot</MenuItem>
              <MenuItem value="TimecoursePNG">TimecoursePNG</MenuItem>
            </Select>
          </FormControl>

          {/* Replot 用 チャンネル */}
          {drawMode === "Replot" && (
            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel id="replot-channel-label">Channel</InputLabel>
              <Select
                labelId="replot-channel-label"
                value={replotChannel}
                label="Channel"
                onChange={(e) => setReplotChannel(e.target.value as "ph" | "fluo1" | "fluo2")}
              >
                <MenuItem value="ph">ph</MenuItem>
                <MenuItem value="fluo1">fluo1</MenuItem>
                <MenuItem value="fluo2">fluo2</MenuItem>
              </Select>
            </FormControl>
          )}

          {/* PixelSD / PixelCV 用 Channel */}
          {(drawMode === "PixelSD" || drawMode === "PixelCV") && (
            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel id="pixelsd-channel-label">Channel</InputLabel>
              <Select
                labelId="pixelsd-channel-label"
                value={pixelSDChannel}
                label="Channel"
                onChange={(e) => setPixelSDChannel(e.target.value as "ph" | "fluo1" | "fluo2")}
              >
                <MenuItem value="ph">ph</MenuItem>
                <MenuItem value="fluo1">fluo1</MenuItem>
                <MenuItem value="fluo2">fluo2</MenuItem>
              </Select>
            </FormControl>
          )}

          {/* TimecoursePNG 用 channel_mode */}
          {drawMode === "TimecoursePNG" && (
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel id="timecourse-channel-label">channel_mode</InputLabel>
              <Select
                labelId="timecourse-channel-label"
                value={timecourseChannelMode}
                label="channel_mode"
                onChange={(e) => setTimecourseChannelMode(e.target.value as ChannelMode)}
              >
                <MenuItem value="ph">ph</MenuItem>
                <MenuItem value="ph_replot">ph_replot</MenuItem>
                <MenuItem value="fluo1">fluo1</MenuItem>
                <MenuItem value="fluo1_replot">fluo1_replot</MenuItem>
                <MenuItem value="fluo2">fluo2</MenuItem>
                <MenuItem value="fluo2_replot">fluo2_replot</MenuItem>
                {/* ▼▼▼ ここが追加 ▼▼▼ */}
                <MenuItem value="all_channels">All</MenuItem>
              </Select>
            </FormControl>
          )}
        </Box>

        {/* メイン表示: カード */}
        {dbName ? (
          <Card sx={{ borderRadius: 2, boxShadow: 2, backgroundColor: "#fff", mb: 4 }}>
            <CardContent>
              <Grid container spacing={2} justifyContent="center" alignItems="flex-start">
                {/* ロード中 */}
                {!allLoaded && (
                  <Grid item xs={12}>
                    <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
                      <CircularProgress />
                      <Typography ml={2}>画像を読み込み中です...</Typography>
                    </Box>
                  </Grid>
                )}

                {allLoaded && (
                  <>
                    {/* 通常GIF */}
                    {normalGifUrls.map((url, idx) => (
                      <Grid item xs={12} md={3} key={`normal-gif-${idx}-${reloadKey}`}>
                        <CardMedia
                          component="img"
                          image={url}
                          alt={`timelapse-${channels[idx]}`}
                          sx={{ width: "100%", borderRadius: 2, objectFit: "contain" }}
                        />
                      </Grid>
                    ))}

                    {/* Replot: 4枚目に Replot GIF */}
                    {drawMode === "Replot" && (
                      <Grid item xs={12} md={3}>
                        <CardMedia
                          component="img"
                          image={replotGifUrl}
                          alt={`replot-${replotChannel}`}
                          key={`replot-${replotChannel}-${reloadKey}`}
                          sx={{ width: "100%", borderRadius: 2, objectFit: "contain" }}
                        />
                      </Grid>
                    )}

                    {/* ContourAreas: グラフ表示 */}
                    {drawMode === "ContourAreas" && (
                      <Grid item xs={12} md={3}>
                        <Box
                          sx={{
                            position: "relative",
                            width: "100%",
                            paddingBottom: "100%",
                            borderRadius: 2,
                            overflow: "hidden",
                          }}
                        >
                          {contourAreas.length > 0 ? (
                            <Box
                              sx={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
                            >
                              <Line
                                data={contourAreasChartData}
                                options={contourAreasChartOptions}
                              />
                            </Box>
                          ) : (
                            <Typography variant="body1" mt={2}>
                              輪郭面積データなし
                            </Typography>
                          )}
                        </Box>
                        <Box mt={1} textAlign="center">
                          <Button
                            variant="contained"
                            size="small"
                            onClick={handleDownloadCsv}
                            startIcon={<DownloadIcon />}
                            sx={{ backgroundColor: "#000", color: "#fff", "&:hover": { backgroundColor: "#333" } }}
                          >
                            Download CSV
                          </Button>
                        </Box>
                      </Grid>
                    )}

                    {/* PixelSD: グラフ表示 */}
                    {drawMode === "PixelSD" && (
                      <Grid item xs={12} md={3}>
                        <Box
                          sx={{
                            position: "relative",
                            width: "100%",
                            paddingBottom: "100%",
                            borderRadius: 2,
                            overflow: "hidden",
                          }}
                        >
                          {pixelSDs.length > 0 ? (
                            <Box sx={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}>
                              <Line data={pixelSDChartData} options={pixelSDChartOptions} />
                            </Box>
                          ) : (
                            <Typography variant="body1" mt={2}>
                              SDデータなし
                            </Typography>
                          )}
                        </Box>
                        <Box mt={1} textAlign="center">
                          <Button
                            variant="contained"
                            size="small"
                            onClick={handleDownloadCsv}
                            startIcon={<DownloadIcon />}
                            sx={{ backgroundColor: "#000", color: "#fff", "&:hover": { backgroundColor: "#333" } }}
                          >
                            Download CSV
                          </Button>
                        </Box>
                      </Grid>
                    )}

                    {/* PixelCV: グラフ表示 */}
                    {drawMode === "PixelCV" && (
                      <Grid item xs={12} md={3}>
                        <Box
                          sx={{
                            position: "relative",
                            width: "100%",
                            paddingBottom: "100%",
                            borderRadius: 2,
                            overflow: "hidden",
                          }}
                        >
                          {pixelCVs.length > 0 ? (
                            <Box sx={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}>
                              <Line data={pixelCVChartData} options={pixelCVChartOptions} />
                            </Box>
                          ) : (
                            <Typography variant="body1" mt={2}>
                              CVデータなし
                            </Typography>
                          )}
                        </Box>
                        <Box mt={1} textAlign="center">
                          <Button
                            variant="contained"
                            size="small"
                            onClick={handleDownloadCsv}
                            startIcon={<DownloadIcon />}
                            sx={{ backgroundColor: "#000", color: "#fff", "&:hover": { backgroundColor: "#333" } }}
                          >
                            Download CSV
                          </Button>
                        </Box>
                      </Grid>
                    )}

                    {/* TimecoursePNG */}
                    {drawMode === "TimecoursePNG" && (
                      <Grid item xs={12}>
                        <CardMedia
                          component="img"
                          image={timecoursePngUrl}
                          alt={`timecourse-${timecourseChannelMode}`}
                          key={`timecourse-png-${reloadKey}`}
                          sx={{
                            width: "100%",
                            borderRadius: 2,
                            objectFit: "contain",
                            maxHeight: "600px",
                          }}
                        />
                      </Grid>
                    )}
                  </>
                )}

                {/* 隠しプレロード <img> で onLoad / onError */}
                {allGifUrls.map((url, idx) => (
                  <img
                    key={`preload-${idx}-${reloadKey}`}
                    src={url}
                    alt="preload"
                    style={{ display: "none" }}
                    onLoad={handleImageLoad}
                    onError={handleImageLoad}
                  />
                ))}
              </Grid>
            </CardContent>
          </Card>
        ) : (
          <Typography variant="body1" mt={2}>
            データベースが指定されていません。
          </Typography>
        )}
      </Container>

      {/* All Cells モーダル */}
      <Dialog open={openModal} onClose={() => setOpenModal(false)} maxWidth="md" fullWidth>
        <DialogTitle>All Cells Preview</DialogTitle>
        <DialogContent>
          {loadingAllCells ? (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
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
