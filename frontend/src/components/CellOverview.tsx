import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import {
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Box,
  Button,
  Typography,
  TextField,
  FormControlLabel,
  Checkbox,
  Breadcrumbs,
  Link,
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { Scatter } from "react-chartjs-2";
import { ChartOptions } from "chart.js";
import Spinner from "./Spinner";
import CellMorphologyTable from "./CellMorphoTable";
import { settings } from "../settings";
import { useSearchParams } from "react-router-dom";
import MedianEngine from "./MedianEngine";
import MeanEngine from "./MeanEngine";
import HeatmapEngine from "./HeatmapEngine";
import VarEngine from "./VarEngine";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

//-----------------------------------
// 型定義
//-----------------------------------
type ImageState = {
  ph: string; // PH画像
  fluo?: string | null; // 蛍光画像 (single_layer系ではnull)
  replot?: string; // Replotグラフ画像
  distribution?: string; // Distribution画像
  distribution_normalized?: string; // Distribution Normalized画像
  path?: string; // Path画像
  prediction?: string; // AI推論画像
  cloud_points?: string; // 3D表示
  cloud_points_ph?: string; // 3D PH表示
};

// 「どのモードにするか」を列挙型的に管理する
type DrawModeType =
  | "light"
  | "replot"
  | "distribution"
  | "distribution_normalized"
  | "path"
  | "t1contour"
  | "prediction"
  | "cloud_points"
  | "cloud_points_ph";

// MorphoEngine関連のモード名
type EngineName =
  | "None"
  | "MorphoEngine 2.0"
  | "MedianEngine"
  | "MeanEngine"
  | "HeatmapEngine"
  | "VarEngine";

// MorphoEngineロゴマッピング
const engineLogos: Record<EngineName, string> = {
  None: "path_to_none_logo.png",
  "MorphoEngine 2.0": "/logo_tp.png",
  MedianEngine: "/logo_dots.png",
  MeanEngine: "/logo_circular.png",
  VarEngine: "/var_logo.png",
  HeatmapEngine: "/logo_heatmap.png",
};

//-----------------------------------
// DetectMode用の型定義
//-----------------------------------
type DetectModeType = "None" | "T1(U-net)" | "Canny" | "Elastic";

//-----------------------------------
// まとめて管理したい設定たち
//-----------------------------------
const url_prefix = settings.url_prefix;

// DrawModeごとに、表示名や「Polyfit Degree入力が必要かどうか」をまとめた設定
const DRAW_MODES: {
  value: DrawModeType;
  label: string;
  needsPolyfit?: boolean;
}[] = [
  { value: "light", label: "Light" },
  { value: "replot", label: "Replot", needsPolyfit: true },
  { value: "distribution", label: "Distribution" },
  { value: "distribution_normalized", label: "Distribution (Normalized)" },
  { value: "path", label: "Peak-path", needsPolyfit: true },
  { value: "t1contour", label: "Light+Model T1" },
  { value: "prediction", label: "Model T1(Torch GPU)" },
  { value: "cloud_points", label: "3D Fluo" },
  { value: "cloud_points_ph", label: "3D PH" },
];

//-----------------------------------
// コンポーネント本体
//-----------------------------------
const CellImageGrid: React.FC = () => {
  // URLクエリから取得
  const [searchParams] = useSearchParams();
  const db_name = searchParams.get("db_name") ?? "test_database.db";
  const cell_number = searchParams.get("cell") ?? "1";
  const init_draw_mode = (searchParams.get("init_draw_mode") ?? "light") as DrawModeType;

  // セルIDや画像などの状態管理
  const [cellIds, setCellIds] = useState<string[]>([]);
  const [images, setImages] = useState<{ [key: string]: ImageState }>({});
  const [selectedLabel, setSelectedLabel] = useState<string>("74");
  const [manualLabel, setManualLabel] = useState<string>("");
  const [currentIndex, setCurrentIndex] = useState<number>(parseInt(cell_number) - 1);

  // 追加: 「今テキストボックスに入力しているインデックス」
  const [inputIndex, setInputIndex] = useState<string>(parseInt(cell_number).toString());

  // 各種スイッチ・入力
  const [drawContour, setDrawContour] = useState<boolean>(true);
  const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
  const [autoPlay, setAutoPlay] = useState<boolean>(false);
  const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
  const [drawMode, setDrawMode] = useState<DrawModeType>(init_draw_mode);
  const [fitDegree, setFitDegree] = useState<number>(4);
  const [engineMode, setEngineMode] = useState<EngineName>("None");

  // DetectMode 用の state
  const [detectMode, setDetectMode] = useState<DetectModeType>("None");
  // 追加: Canny の閾値を入力するための state（例: Threshold2）
  const [cannyThresh2, setCannyThresh2] = useState<number>(100);
  // Elastic deformation parameter
  const [elasticDelta, setElasticDelta] = useState<number>(0);

  // 読み込み状態や輪郭データなど
  const [isLoading, setIsLoading] = useState(false);
  const [contourData, setContourData] = useState<number[][]>([]);
  const [contourDataT1, setContourDataT1] = useState<number[][]>([]);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

  // 連続送り用のデバウンス管理
  const lastCallTimeRef = useRef<number | null>(null);
  const debounce = (func: () => void, wait: number) => {
    return () => {
      const now = new Date().getTime();
      if (lastCallTimeRef.current === null || now - lastCallTimeRef.current > wait) {
        lastCallTimeRef.current = now;
        func();
      }
    };
  };

  //------------------------------------
  // セル一覧の取得
  //------------------------------------
  useEffect(() => {
    const fetchCellIds = async () => {
      const response = await axios.get(`${url_prefix}/cells/${db_name}/${selectedLabel}`);
      const ids = response.data.map((cell: { cell_id: string }) => cell.cell_id);
      setCellIds(ids);
    };
    fetchCellIds();
  }, [db_name, selectedLabel]);

  //------------------------------------
  // 輪郭データの取得
  //------------------------------------
  const fetchContour = async (cellId: string) => {
    try {
      const response = await axios.get(`${url_prefix}/cells/${cellId}/contour/raw?db_name=${db_name}`);
      setContourData(response.data.contour);
    } catch (error) {
      console.error("Error fetching contour data:", error);
    }
  };

  const fetchContourT1 = async (cellId: string) => {
    try {
      const response = await axios.get(`${url_prefix}/cell_ai/${db_name}/${cellId}/plot_data`);
      setContourDataT1(response.data);
    } catch (error) {
      console.error("Error fetching contour data:", error);
    }
  };

  //------------------------------------
  // 画像の一括フェッチ用: PH, Fluo
  //------------------------------------
  const fetchStandardImages = async (cellId: string) => {
    try {
      // PH画像取得
      const phUrl = await fetchImage("ph_image", cellId, db_name);
      // 蛍光画像取得 (single_layer系は蛍光画像が無いのでスキップ)
      let fluoUrl: string | null = null;
      if (!db_name.includes("single_layer")) {
        fluoUrl = await fetchImage("fluo_image", cellId, db_name, brightnessFactor);
      }

      // ステート更新
      setImages((prev) => ({
        ...prev,
        [cellId]: {
          ...prev[cellId],
          ph: phUrl,
          fluo: fluoUrl,
        },
      }));

      // 画像サイズを保存 (PH画像を基準)
      const dim = await getImageDimensions(phUrl);
      setImageDimensions(dim);
    } catch (error) {
      console.error("Error fetching standard images:", error);
    }
  };

  //------------------------------------
  // 通常画像フェッチ用のユーティリティ
  //------------------------------------
  const fetchImage = async (
    type: "ph_image" | "fluo_image",
    cellId: string,
    dbName: string,
    brightness: number = 1.0
  ): Promise<string> => {
    let url = `${url_prefix}/cells/${cellId}/${dbName}/${drawContour}/${drawScaleBar}/`;
    if (type === "fluo_image") {
      // 蛍光画像は明るさ係数をクエリで追加
      url += `${type}?brightness_factor=${brightness}`;
    } else {
      url += `${type}`;
    }
    console.log(`Fetching image from URL: ${url}`);
    const response = await axios.get(url, { responseType: "blob" });
    return URL.createObjectURL(response.data);
  };

  //------------------------------------
  // 画像の寸法を取得するユーティリティ
  //------------------------------------
  const getImageDimensions = (imageUrl: string): Promise<{ width: number; height: number }> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.width, height: img.height });
      img.onerror = reject;
      img.src = imageUrl;
    });
  };

  //------------------------------------
  // drawModeの追加画像を取得
  //------------------------------------
  const fetchAdditionalImage = async (mode: DrawModeType, cellId: string) => {
    try {
      switch (mode) {
        case "replot": {
          if (!images[cellId]?.replot) {
            const response = await axios.get(
              `${url_prefix}/cells/${cellId}/${db_name}/replot?degree=${fitDegree}`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], replot: url },
            }));
          }
          break;
        }
        case "distribution": {
          if (!images[cellId]?.distribution) {
            const response = await axios.get(
              `${url_prefix}/cells/${db_name}/${cellId}/distribution`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], distribution: url },
            }));
          }
          break;
        }
        case "distribution_normalized": {
          if (!images[cellId]?.distribution_normalized) {
            const response = await axios.get(
              `${url_prefix}/cells/${db_name}/${selectedLabel}/${cellId}/distribution_normalized`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], distribution_normalized: url },
            }));
          }
          break;
        }
        case "path": {
          if (!images[cellId]?.path) {
            setIsLoading(true);
            const response = await axios.get(
              `${url_prefix}/cells/${cellId}/${db_name}/path?degree=${fitDegree}`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], path: url },
            }));
            setIsLoading(false);
          }
          break;
        }
        case "prediction": {
          if (!images[cellId]?.prediction) {
            const response = await axios.get(`${url_prefix}/cell_ai/${db_name}/${cellId}`, {
              responseType: "blob",
            });
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], prediction: url },
            }));
          }
          break;
        }
        case "cloud_points": {
          if (!images[cellId]?.cloud_points) {
            const response = await axios.get(
              `${url_prefix}/cells/${db_name}/${cellId}/3d`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], cloud_points: url },
            }));
          }
          break;
        }
        case "cloud_points_ph": {
          if (!images[cellId]?.cloud_points_ph) {
            const response = await axios.get(
              `${url_prefix}/cells/${db_name}/${cellId}/3d-ph`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], cloud_points_ph: url },
            }));
          }
          break;
        }
        case "t1contour":
        default:
          break;
      }
    } catch (error) {
      console.error(`Error fetching additional image for mode: ${mode}`, error);
      setIsLoading(false);
    }
  };

  //------------------------------------
  // セルを切り替えたときのメイン処理
  //------------------------------------
  useEffect(() => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];
    fetchStandardImages(cellId);
    fetchContour(cellId);
    fetchContourT1(cellId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cellIds, currentIndex, db_name, drawContour, drawScaleBar, brightnessFactor]);

  //------------------------------------
  // drawModeが変わったら追加画像をフェッチ
  //------------------------------------
  useEffect(() => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];
    fetchAdditionalImage(drawMode, cellId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawMode, cellIds, currentIndex, fitDegree]);

  //------------------------------------
  // 現在のセルIDに対応した初期ラベルを取得
  //------------------------------------
  useEffect(() => {
    const fetchInitialLabel = async () => {
      if (cellIds.length > 0) {
        const cellId = cellIds[currentIndex];
        try {
          const response = await axios.get(`${url_prefix}/cells/${db_name}/${cellId}/label`);
          const label = response.data.toString();
          console.log(`Initial label for cell ${cellId}: ${label}`);
          setManualLabel(label);
        } catch (error) {
          console.error("Error fetching initial label:", error);
        }
      }
    };
    fetchInitialLabel();
  }, [cellIds, currentIndex, db_name]);

  //------------------------------------
  // Next/Prev/AutoPlay
  //------------------------------------
  const handleNext = debounce(() => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % cellIds.length);
  }, 500);

  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex - 1 + cellIds.length) % cellIds.length);
  };

  useEffect(() => {
    let autoNextInterval: NodeJS.Timeout | undefined;
    if (autoPlay) {
      autoNextInterval = setInterval(handleNext, 3000); // 3秒ごとに自動でNext
    }
    return () => {
      if (autoNextInterval) clearInterval(autoNextInterval);
    };
  }, [autoPlay, handleNext]);

  //------------------------------------
  // Labelセレクト変更
  //------------------------------------
  const handleLabelChange = (event: SelectChangeEvent<string>) => {
    setSelectedLabel(event.target.value);
  };

  const handleCellLabelChange = async (event: SelectChangeEvent<string>) => {
    const newLabel = event.target.value;
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];

    try {
      await axios.patch(`${url_prefix}/cells/${db_name}/${cellId}/${newLabel}`);
      setManualLabel(newLabel);
    } catch (error) {
      console.error("Error updating cell label:", error);
    }
  };

  //------------------------------------
  // Detect Modeセレクト変更
  //------------------------------------
  const handleDetectModeChange = (event: SelectChangeEvent<string>) => {
    setDetectMode(event.target.value as DetectModeType);
  };

  //------------------------------------
  // T1(U-net) Detectボタン押下時のハンドラ
  //------------------------------------
  const handleT1Detect = async () => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];

    try {
      // PATCHリクエスト送信
      const patchUrl = `${url_prefix}/cells/redetect_contour_t1/${db_name}/${cellId}`;
      console.log("Calling patch:", patchUrl);
      await axios.patch(patchUrl);

      // 成功したら輪郭・画像を再取得
      await fetchContour(cellId);
      await fetchContourT1(cellId);
      await fetchStandardImages(cellId); // ← PH / Fluo の再取得
    } catch (err) {
      console.error("Error calling T1 detect:", err);
    }
  };

  //------------------------------------
  // Canny Detectボタン押下時のハンドラ
  //------------------------------------
  const handleCannyDetect = async () => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];

    try {
      // PATCHリクエスト送信（クエリに canny_thresh2 を付与）
      const patchUrl = `${url_prefix}/cells/redetect_contour_canny/${db_name}/${cellId}?canny_thresh2=${cannyThresh2}`;
      console.log("Calling patch:", patchUrl);
      await axios.patch(patchUrl);

      // 成功したら輪郭・画像を再取得
      await fetchContour(cellId);
      await fetchStandardImages(cellId); // ← PH / Fluo の再取得
    } catch (err) {
      console.error("Error calling Canny detect:", err);
    }
  };

  //------------------------------------
  // Elastic Apply ボタン押下時のハンドラ
  //------------------------------------
  const handleElasticApply = async () => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];
    try {
      const patchUrl = `${url_prefix}/cells/elastic_contour/${db_name}/${cellId}?delta=${elasticDelta}`;
      await axios.patch(patchUrl);
      await fetchContour(cellId);
      await fetchStandardImages(cellId);
    } catch (err) {
      console.error("Error calling Elastic apply:", err);
    }
  };

  //------------------------------------
  // その他ハンドラ
  //------------------------------------
  const handleContourChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDrawContour(e.target.checked);
  };
  const handleScaleBarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDrawScaleBar(e.target.checked);
  };
  const handleAutoPlayChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setAutoPlay(e.target.checked);
  };
  const handleBrightnessChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBrightnessFactor(parseFloat(e.target.value));
  };
  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    event.currentTarget.blur();
    event.preventDefault();
  };
  const handleDrawModeChange = (event: SelectChangeEvent<string>) => {
    setDrawMode(event.target.value as DrawModeType);
  };
  const handleFitDegreeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFitDegree(parseInt(e.target.value, 10));
  };
  const handleEngineModeChange = (event: SelectChangeEvent<string>) => {
    setEngineMode(event.target.value as EngineName);
  };

  //------------------------------------
  // キーボードイベント (EnterキーでNextなど)
  //------------------------------------
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        handleNext();
      } else if (["1", "2", "3", "n"].includes(event.key)) {
        let newLabel = event.key;
        if (newLabel === "n") {
          newLabel = "1000"; // 'n' は "1000" に対応
        }
        setManualLabel(newLabel);
        handleCellLabelChange({ target: { value: newLabel } } as SelectChangeEvent<string>);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cellIds, currentIndex]);

  //------------------------------------
  // Contour用データセット
  //------------------------------------
  const contourPlotData = {
    datasets: [
      {
        label: "Contour",
        data: contourData.map((point) => ({ x: point[0], y: point[1] })),
        borderColor: "lime",
        backgroundColor: "lime",
        pointRadius: 1,
      },
      ...(drawMode === "t1contour"
        ? [
            {
              label: "Model T1",
              data: contourDataT1.map((point) => ({ x: point[0], y: point[1] })),
              borderColor: "red",
              backgroundColor: "red",
              pointRadius: 1,
            },
          ]
        : []),
    ],
  };

  //------------------------------------
  // Contour用オプション
  //------------------------------------
  const contourPlotOptions: ChartOptions<"scatter"> = {
    maintainAspectRatio: true,
    aspectRatio: 1,
    scales: {
      x: {
        type: "linear",
        position: "bottom",
        min: 0,
        max: imageDimensions?.width,
        reverse: false,
      },
      y: {
        type: "linear",
        min: 0,
        max: imageDimensions?.height,
        reverse: true,
      },
    },
  };

  //------------------------------------
  // 「テキスト入力からインデックスをジャンプ」するハンドラ
  //------------------------------------
  const handleIndexJump = () => {
    const newIndex = parseInt(inputIndex, 10) - 1;
    if (!isNaN(newIndex) && newIndex >= 0 && newIndex < cellIds.length) {
      setCurrentIndex(newIndex);
    } else {
      console.warn("Invalid index:", newIndex);
    }
  };

  // currentIndex が変わったら、テキスト入力も追随しておく
  useEffect(() => {
    setInputIndex((currentIndex + 1).toString());
  }, [currentIndex]);

  //------------------------------------
  // 実際の描画部分
  //------------------------------------
  return (
    <Box sx={{ p: { xs: 2, md: 3 }, width: "100%" }}>
      {/* パンくずリスト */}
      <Box mb={2}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Link underline="hover" color="inherit" href="/dbconsole">
            Database Console
          </Link>
          <Typography color="text.primary">{db_name}</Typography>
        </Breadcrumbs>
      </Box>

      {/* 全体レイアウトをGridに変えてレスポンシブ対応 */}
      <Grid container spacing={3}>
        {/* ----- 左カラム: PH/Fluoや、Prev/Nextなど ----- */}
        <Grid item xs={12} lg={5}>
          <Box sx={{ mb: 3 }}>
            {/* 
              Label / DetectMode / (CannyThresh2) / Detect ボタン
              を同じ行に表示しつつ、DetectModeが None のときは余白を埋める 
            */}
            <Grid container spacing={2} alignItems="center">
              {/* Label選択 */}
              <Grid item xs={3}>
                <FormControl fullWidth variant="outlined">
                  <InputLabel id="label-select-label">Label</InputLabel>
                  <Select
                    labelId="label-select-label"
                    label="Label"
                    value={selectedLabel}
                    onChange={handleLabelChange}
                  >
                    <MenuItem value="74">All</MenuItem>
                    <MenuItem value="1000">N/A</MenuItem>
                    <MenuItem value="1">1</MenuItem>
                    <MenuItem value="2">2</MenuItem>
                    <MenuItem value="3">3</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              {/* DetectMode が None のときは xs=9 を割り当て、そうでなければ xs=3 */}
              {detectMode === "None" ? (
                <Grid item xs={9}>
                  <FormControl fullWidth variant="outlined">
                    <InputLabel id="detect-mode-select-label">Detect Mode</InputLabel>
                    <Select
                      labelId="detect-mode-select-label"
                      label="Detect Mode"
                      value={detectMode}
                      onChange={handleDetectModeChange}
                    >
                      <MenuItem value="None">None</MenuItem>
                      <MenuItem value="T1(U-net)">T1(U-net)</MenuItem>
                      <MenuItem value="Canny">Canny</MenuItem>
                      <MenuItem value="Elastic">Elastic</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              ) : (
                <>
                  {/* DetectMode が None 以外の場合は xs=3 */}
                  <Grid item xs={3}>
                    <FormControl fullWidth variant="outlined">
                      <InputLabel id="detect-mode-select-label">Detect Mode</InputLabel>
                      <Select
                        labelId="detect-mode-select-label"
                        label="Detect Mode"
                        value={detectMode}
                        onChange={handleDetectModeChange}
                      >
                        <MenuItem value="None">None</MenuItem>
                        <MenuItem value="T1(U-net)">T1(U-net)</MenuItem>
                        <MenuItem value="Canny">Canny</MenuItem>
                        <MenuItem value="Elastic">Elastic</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>

                  {/* detectMode が "Canny" の場合だけ CannyThresh2 を表示 */}
                  {detectMode === "Canny" && (
                    <Grid item xs={3}>
                      <TextField
                        label="Canny Th2"
                        variant="outlined"
                        type="number"
                        size="small"
                        value={cannyThresh2}
                        onChange={(e) => setCannyThresh2(parseInt(e.target.value, 10))}
                        InputProps={{
                          onWheel: handleWheel,
                        }}
                        fullWidth
                      />
                    </Grid>
                  )}
                  {detectMode === "Elastic" && (
                    <Grid item xs={3}>
                      <TextField
                        label="Elastic \u0394"
                        variant="outlined"
                        type="number"
                        size="small"
                        value={elasticDelta}
                        onChange={(e) => setElasticDelta(parseInt(e.target.value, 10))}
                        InputProps={{
                          inputProps: { min: -3, max: 3, step: 1 },
                          onWheel: handleWheel,
                        }}
                        fullWidth
                      />
                    </Grid>
                  )}

                  {/* Detect ボタン:
                      detectMode が "T1(U-net)" のとき → T1Detect,
                      detectMode が "Canny" のとき → CannyDetect
                  */}
                  <Grid
                    item
                    xs={detectMode === "Canny" || detectMode === "Elastic" ? 3 : 6}
                  >
                    {detectMode === "T1(U-net)" && (
                      <Button
                        variant="contained"
                        color="secondary"
                        onClick={handleT1Detect}
                        fullWidth
                      >
                        Detect
                      </Button>
                    )}
                    {detectMode === "Canny" && (
                      <Button
                        variant="contained"
                        color="secondary"
                        onClick={handleCannyDetect}
                        fullWidth
                      >
                        Detect
                      </Button>
                    )}
                    {detectMode === "Elastic" && (
                      <Button
                        variant="contained"
                        color="secondary"
                        onClick={handleElasticApply}
                        fullWidth
                      >
                        Apply
                      </Button>
                    )}
                  </Grid>
                </>
              )}
            </Grid>
          </Box>

          {/* チェックボックス類 */}
          <Box sx={{ mb: 3 }}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={2}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={drawContour}
                      onChange={handleContourChange}
                      style={{ color: "black" }}
                    />
                  }
                  label="Contour"
                  style={{ color: "black" }}
                />
              </Grid>
              <Grid item xs={2}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={drawScaleBar}
                      onChange={handleScaleBarChange}
                      style={{ color: "black" }}
                    />
                  }
                  label="Scale"
                  style={{ color: "black" }}
                />
              </Grid>
              <Grid item xs={2}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={autoPlay}
                      onChange={handleAutoPlayChange}
                      style={{ color: "black" }}
                    />
                  }
                  label="Auto"
                  style={{ color: "black" }}
                />
              </Grid>
              <Grid item xs={3}>
                <TextField
                  label="Brightness"
                  variant="outlined"
                  type="number"
                  value={brightnessFactor}
                  onChange={handleBrightnessChange}
                  InputProps={{
                    inputProps: { min: 0.1, step: 0.1 },
                    onWheel: handleWheel,
                    autoComplete: "off",
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={3}>
                <FormControl fullWidth variant="outlined">
                  <InputLabel id="manual-label-select-label">Manual Label</InputLabel>
                  <Select
                    labelId="manual-label-select-label"
                    label="Manual Label"
                    value={manualLabel}
                    onChange={handleCellLabelChange}
                  >
                    <MenuItem value="1000">N/A</MenuItem>
                    <MenuItem value="1">1</MenuItem>
                    <MenuItem value="2">2</MenuItem>
                    <MenuItem value="3">3</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>

          {/* Prev / Next */}
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            sx={{ mb: 3 }}
          >
            <Button
              variant="contained"
              color="primary"
              onClick={handlePrev}
              disabled={cellIds.length === 0}
              style={{ backgroundColor: "black", minWidth: "100px" }}
            >
              Prev
            </Button>

            {/* テキストボックスに入力してセル番号ジャンプ */}
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="h6" component="span">
                Cell
              </Typography>
              <TextField
                type="number"
                variant="outlined"
                size="small"
                value={inputIndex}
                onChange={(e) => setInputIndex(e.target.value)}
                onBlur={handleIndexJump}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleIndexJump();
                  }
                }}
                style={{ width: 60 }}
              />
              <Typography variant="h6" component="span">
                of {cellIds.length}
              </Typography>
              <Typography variant="h6" component="span">
                / {cellIds[currentIndex]}
              </Typography>
            </Box>

            <Button
              variant="contained"
              color="primary"
              onClick={handleNext}
              disabled={cellIds.length === 0}
              style={{ backgroundColor: "black", minWidth: "100px" }}
            >
              Next
            </Button>
          </Box>

          {/* PH / Fluo画像 */}
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              {images[cellIds[currentIndex]] ? (
                <img
                  src={images[cellIds[currentIndex]].ph}
                  alt={`Cell ${cellIds[currentIndex]} PH`}
                  style={{ width: "100%", height: "auto" }}
                />
              ) : (
                <div>Loading PH...</div>
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              {images[cellIds[currentIndex]] && images[cellIds[currentIndex]].fluo ? (
                <img
                  src={images[cellIds[currentIndex]].fluo as string}
                  alt={`Cell ${cellIds[currentIndex]} Fluo`}
                  style={{ width: "100%", height: "auto" }}
                />
              ) : db_name.includes("single_layer") ? (
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    height: "100%",
                  }}
                >
                  <Typography variant="h5">Single layer mode.</Typography>
                  <img
                    src="/logo_dots.png"
                    alt="Morpho Engine is off"
                    style={{ maxWidth: "15%", maxHeight: "15%" }}
                  />
                </Box>
              ) : (
                <div>Not available</div>
              )}
            </Grid>
          </Grid>
        </Grid>

        {/* ----- 中央カラム: DrawModeごとの追加表示 ----- */}
        <Grid item xs={12} md={6} lg={4}>
          <Box sx={{ mb: 2 }}>
            {/* DrawMode セレクト + Polyfit Degree */}
            <FormControl fullWidth variant="outlined">
              <InputLabel id="draw-mode-select-label">Draw Mode</InputLabel>
              <Select
                labelId="draw-mode-select-label"
                label="Draw Mode"
                value={drawMode}
                onChange={handleDrawModeChange}
              >
                {DRAW_MODES.map((mode) => (
                  <MenuItem key={mode.value} value={mode.value}>
                    {mode.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {DRAW_MODES.find((m) => m.value === drawMode)?.needsPolyfit && (
              <Box mt={2}>
                <TextField
                  label="Polyfit Degree"
                  variant="outlined"
                  type="number"
                  value={fitDegree}
                  onChange={handleFitDegreeChange}
                  InputProps={{
                    inputProps: { min: 0, step: 1 },
                    onWheel: handleWheel,
                    autoComplete: "off",
                  }}
                  fullWidth
                />
              </Box>
            )}
          </Box>

          <Box sx={{ mb: 2 }}>
            {/* モード別の表示 */}
            {drawMode === "light" && (
              <Scatter data={contourPlotData} options={contourPlotOptions} />
            )}
            {drawMode === "t1contour" && (
              <Scatter data={contourPlotData} options={contourPlotOptions} />
            )}

            {drawMode === "replot" && images[cellIds[currentIndex]]?.replot && (
              <img
                src={images[cellIds[currentIndex]]?.replot}
                alt={`Cell ${cellIds[currentIndex]} Replot`}
                style={{ width: "100%" }}
              />
            )}

            {drawMode === "distribution" &&
              images[cellIds[currentIndex]]?.distribution && (
                <img
                  src={images[cellIds[currentIndex]]?.distribution}
                  alt={`Cell ${cellIds[currentIndex]} Distribution`}
                  style={{ width: "100%" }}
                />
              )}

            {drawMode === "distribution_normalized" &&
              images[cellIds[currentIndex]]?.distribution_normalized && (
                <img
                  src={images[cellIds[currentIndex]]?.distribution_normalized}
                  alt={`Cell ${cellIds[currentIndex]} Distribution (Normalized)`}
                  style={{ width: "100%" }}
                />
              )}

            {drawMode === "path" && isLoading ? (
              <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                style={{ height: 400 }}
              >
                <Spinner />
              </Box>
            ) : (
              drawMode === "path" &&
              images[cellIds[currentIndex]]?.path && (
                <img
                  src={images[cellIds[currentIndex]]?.path}
                  alt={`Cell ${cellIds[currentIndex]} Path`}
                  style={{ width: "100%" }}
                />
              )
            )}

            {drawMode === "prediction" &&
              images[cellIds[currentIndex]]?.prediction && (
                <img
                  src={images[cellIds[currentIndex]]?.prediction}
                  alt={`Cell ${cellIds[currentIndex]} Prediction`}
                  style={{ width: "100%" }}
                />
              )}

            {drawMode === "cloud_points" &&
              images[cellIds[currentIndex]]?.cloud_points && (
                <img
                  src={images[cellIds[currentIndex]]?.cloud_points}
                  alt={`Cell ${cellIds[currentIndex]} 3D`}
                  style={{ width: "100%" }}
                />
              )}

            {drawMode === "cloud_points_ph" &&
              images[cellIds[currentIndex]]?.cloud_points_ph && (
                <img
                  src={images[cellIds[currentIndex]]?.cloud_points_ph}
                  alt={`Cell ${cellIds[currentIndex]} 3D`}
                  style={{ width: "100%" }}
                />
              )}
          </Box>
        </Grid>

        {/* ----- 右カラム: MorphoEngine (None, Median, Mean, Var, Heatmap) ----- */}
        <Grid item xs={12} md={6} lg={3}>
          <Box sx={{ mb: 2 }}>
            <FormControl fullWidth variant="outlined">
              <InputLabel id="engine-select-label">MorphoEngine</InputLabel>
              <Select
                labelId="engine-select-label"
                label="MorphoEngine"
                id="engine-select"
                value={engineMode}
                onChange={handleEngineModeChange}
                renderValue={(selected: string) => {
                  if (selected === "None") {
                    return "None";
                  } else {
                    const engineName = selected as EngineName;
                    let displayText: string = engineName;
                    return (
                      <Box display="flex" alignItems="center">
                        {engineName !== "None" && (
                          <img
                            src={engineLogos[engineName]}
                            alt=""
                            style={{ width: 24, height: 24, marginRight: 8 }}
                          />
                        )}
                        {displayText}
                      </Box>
                    );
                  }
                }}
              >
                {Object.entries(engineLogos).map(([engine, logoPath]) => (
                  <MenuItem key={engine} value={engine}>
                    <Box display="flex" alignItems="center">
                      {engine !== "None" && (
                        <img
                          src={logoPath}
                          alt=""
                          style={{ width: 24, height: 24, marginRight: 8 }}
                        />
                      )}
                      {engine}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>

          {engineMode === "None" && (
            <Box
              sx={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                height: "100%",
                mt: 3,
              }}
            >
              <Typography variant="h5">MorphoEngine is off.</Typography>
              <img
                src="/logo_tp.png"
                alt="Morpho Engine is off"
                style={{ maxWidth: "15%", maxHeight: "15%", marginLeft: 8 }}
              />
            </Box>
          )}

          {engineMode === "MorphoEngine 2.0" && (
            <Box mt={2}>
              <CellMorphologyTable
                cellId={cellIds[currentIndex]}
                db_name={db_name}
                polyfitDegree={fitDegree}
              />
            </Box>
          )}

          {engineMode === "MedianEngine" && (
            <Box mt={6}>
              <MedianEngine
                dbName={db_name}
                label={selectedLabel}
                cellId={cellIds[currentIndex]}
              />
            </Box>
          )}

          {engineMode === "MeanEngine" && (
            <Box mt={6}>
              <MeanEngine
                dbName={db_name}
                label={selectedLabel}
                cellId={cellIds[currentIndex]}
              />
            </Box>
          )}

          {engineMode === "VarEngine" && (
            <Box mt={6}>
              <VarEngine
                dbName={db_name}
                label={selectedLabel}
                cellId={cellIds[currentIndex]}
              />
            </Box>
          )}

          {engineMode === "HeatmapEngine" && (
            <Box mt={6}>
              <HeatmapEngine
                dbName={db_name}
                label={selectedLabel}
                cellId={cellIds[currentIndex]}
                degree={fitDegree}
              />
            </Box>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default CellImageGrid;
