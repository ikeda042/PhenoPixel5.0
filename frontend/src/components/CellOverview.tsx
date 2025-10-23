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
  IconButton,
  Tooltip,
  Snackbar,
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { Scatter } from "react-chartjs-2";
import { ChartOptions } from "chart.js";
import { useTheme } from "@mui/material/styles";
import Spinner from "./Spinner";
import CellMorphologyTable from "./CellMorphoTable";
import { settings } from "../settings";
import { useSearchParams } from "react-router-dom";
import MedianEngine from "./MedianEngine";
import MeanEngine from "./MeanEngine";
import HeatmapEngine from "./HeatmapEngine";
import VarEngine from "./VarEngine";
import AreaFractionEngine from "./AreaFractionEngine";
import SDEngine from "./SDEngine";
import PixelCVEngine from "./PixelCVEngine";
import PixelEngine from "./PixelEngine";
import IbpAEngine from "./IbpAEngine";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
} from "chart.js";
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend
);

//-----------------------------------
// 型定義
//-----------------------------------
type ImageState = {
  ph: string; // PH画像
  fluo?: string | null; // 蛍光画像 (single_layer系ではnull)
  fluo2?: string | null; // 蛍光画像2
  replot?: string; // Replotグラフ画像
  distribution?: string; // Distribution画像
  distribution_normalized?: string; // Distribution Normalized画像
  path?: string; // Path画像
  prediction?: string; // AI推論画像
  cloud_points?: string; // 3D表示
  cloud_points_ph?: string; // 3D PH表示
  laplacian?: string; // Laplacian画像
  sobel?: string; // Sobel画像
  hu_mask?: string; // HU Mask画像
  map256?: string; // Map256画像
  map256_jet?: string; // Map256 Jet画像
  map256_clip?: string; // Map256 clipped image
};

// 「どのモードにするか」を列挙型的に管理する
type DrawModeType =
  | "light"
  | "replot"
  | "distribution"
  | "distribution_normalized"
  | "path"
  | "laplacian"
  | "sobel"
  | "hu_mask"
  | "map256"
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
  | "VarEngine"
  | "AreaFractionEngine"
  | "SDEngine"
  | "PixelCVEngine"
  | "PixelEngine"
  | "IbpAEngine";

// MorphoEngineロゴマッピング
const engineLogos: Record<EngineName, string> = {
  None: "path_to_none_logo.png",
  "MorphoEngine 2.0": "/logo_tp.png",
  MedianEngine: "/logo_dots.png",
  MeanEngine: "/logo_circular.png",
  VarEngine: "/var_logo.png",
  AreaFractionEngine: "/logo_cross.png",
  HeatmapEngine: "/logo_heatmap.png",
  SDEngine: "/var_logo.png",
  PixelCVEngine: "/var_logo.png",
  PixelEngine: "/var_logo.png",
  IbpAEngine: "/var_logo.png",
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
  { value: "laplacian", label: "Laplacian" },
  { value: "sobel", label: "Sobel" },
  { value: "hu_mask", label: "HU Mask" },
  { value: "map256", label: "Map256", needsPolyfit: true },
  { value: "t1contour", label: "Light+Model T1" },
  { value: "prediction", label: "Model T1(Torch GPU)" },
  { value: "cloud_points", label: "3D Fluo" },
  { value: "cloud_points_ph", label: "3D PH" },
];

const LABEL_OPTIONS: { value: string; label: string }[] = [
  { value: "74", label: "All" },
  { value: "1000", label: "N/A" },
  { value: "1", label: "1" },
  { value: "2", label: "2" },
  { value: "3", label: "3" },
];

//-----------------------------------
// コンポーネント本体
//-----------------------------------
const CellImageGrid: React.FC = () => {
  // URLクエリから取得
  const [searchParams, setSearchParams] = useSearchParams();
  const dbQueryParam = searchParams.get("db") ?? searchParams.get("db_name");
  const db_name = dbQueryParam ?? "test_database.db";
  const initialCellIdRef = useRef<string | null>(searchParams.get("cell_id"));
  const cell_number_param = searchParams.get("cell") ?? "1";
  const initialIndex = (() => {
    const parsed = parseInt(cell_number_param, 10);
    if (isNaN(parsed)) {
      return 0;
    }
    return Math.max(parsed - 1, 0);
  })();
  const drawModeParam = (searchParams.get("draw_mode")
    ?? searchParams.get("init_draw_mode")
    ?? "light") as DrawModeType;
  const theme = useTheme();

  // セルIDや画像などの状態管理
  const [cellIds, setCellIds] = useState<string[]>([]);
  const [images, setImages] = useState<{ [key: string]: ImageState }>({});
  const [selectedLabel, setSelectedLabel] = useState<string>("74");
  const [manualLabel, setManualLabel] = useState<string>("");
  const [currentIndex, setCurrentIndex] = useState<number>(initialIndex);

  // 追加: 「今テキストボックスに入力しているインデックス」
  const [inputIndex, setInputIndex] = useState<string>((initialIndex + 1).toString());

  // 各種スイッチ・入力
  const [drawContour, setDrawContour] = useState<boolean>(true);
  const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
  const [autoPlay, setAutoPlay] = useState<boolean>(false);
  const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
  const [laplacianBrightness, setLaplacianBrightness] = useState<number>(1.0);
  const [sobelBrightness, setSobelBrightness] = useState<number>(1.0);
  const [hasFluo2, setHasFluo2] = useState<boolean>(false);
  const [fluoChannel, setFluoChannel] = useState<'fluo1' | 'fluo2'>('fluo1');
  const [drawMode, setDrawMode] = useState<DrawModeType>(drawModeParam);
  const [fitDegree, setFitDegree] = useState<number>(4);
  const [map256Source, setMap256Source] = useState<'ph' | 'fluo1' | 'fluo2'>('fluo1');
  const [statSource, setStatSource] = useState<'ph' | 'fluo1' | 'fluo2'>('fluo1');
  const [engineMode, setEngineMode] = useState<EngineName>("None");
  const [copyMessage, setCopyMessage] = useState<string | null>(null);

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

  useEffect(() => {
    if (!initialCellIdRef.current || cellIds.length === 0) {
      return;
    }
    const targetIndex = cellIds.findIndex((id) => id === initialCellIdRef.current);
    if (targetIndex !== -1) {
      setCurrentIndex(targetIndex);
    }
    initialCellIdRef.current = null;
  }, [cellIds]);

  useEffect(() => {
    if (cellIds.length === 0) {
      return;
    }
    if (currentIndex >= cellIds.length) {
      setCurrentIndex(cellIds.length - 1);
    }
  }, [cellIds, currentIndex]);

  useEffect(() => {
    if (cellIds.length === 0) {
      return;
    }
    const currentCellId = cellIds[currentIndex];
    if (!currentCellId) {
      return;
    }
    const params = new URLSearchParams(searchParams);
    let shouldUpdate = false;

    const ensureValue = (key: string, value: string) => {
      if (params.get(key) !== value) {
        params.set(key, value);
        shouldUpdate = true;
      }
    };

    const ensureOrRemove = (key: string, value: string | null) => {
      if (value === null) {
        if (params.has(key)) {
          params.delete(key);
          shouldUpdate = true;
        }
        return;
      }
      ensureValue(key, value);
    };

    ensureValue("db", db_name);
    ensureValue("db_name", db_name);
    ensureValue("cell", (currentIndex + 1).toString());
    ensureValue("cell_id", currentCellId);
    ensureValue("draw_mode", drawMode);

    const requiresPolyfit = DRAW_MODES.find((mode) => mode.value === drawMode)?.needsPolyfit ?? false;
    ensureOrRemove(
      "fit_degree",
      requiresPolyfit ? fitDegree.toString() : null,
    );

    if (shouldUpdate) {
      setSearchParams(params, { replace: true });
    }
  }, [
    cellIds,
    currentIndex,
    db_name,
    drawMode,
    fitDegree,
    searchParams,
    setSearchParams,
  ]);

  useEffect(() => {
    const modeFromParams = (searchParams.get("draw_mode")
      ?? searchParams.get("init_draw_mode")
      ?? "light") as DrawModeType;
    if (DRAW_MODES.some((mode) => mode.value === modeFromParams)) {
      setDrawMode((prev) => (prev === modeFromParams ? prev : modeFromParams));
    }

    const fitDegreeParam = searchParams.get("fit_degree");
    if (fitDegreeParam) {
      const parsedDegree = parseInt(fitDegreeParam, 10);
      if (!Number.isNaN(parsedDegree)) {
        setFitDegree((prev) => (prev === parsedDegree ? prev : parsedDegree));
      }
    }
  }, [searchParams]);

  useEffect(() => {
    const checkFluo2 = async () => {
      try {
        const res = await axios.get(`${url_prefix}/databases/${db_name}/has-fluo2`);
        setHasFluo2(res.data.has_fluo2);
      } catch (e) {
        console.error('Failed to check fluo2 existence', e);
      }
    };
    checkFluo2();
  }, [db_name]);

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
      let fluo2Url: string | null = null;
      if (!db_name.includes("single_layer")) {
        fluoUrl = await fetchImage("fluo_image", cellId, db_name, brightnessFactor);
        if (hasFluo2) {
          fluo2Url = await fetchImage("fluo2_image", cellId, db_name, brightnessFactor);
        }
      }

      // ステート更新
      setImages((prev) => ({
        ...prev,
        [cellId]: {
          ...prev[cellId],
          ph: phUrl,
          fluo: fluoUrl,
          fluo2: fluo2Url,
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
    type: "ph_image" | "fluo_image" | "fluo2_image",
    cellId: string,
    dbName: string,
    brightness: number = 1.0
  ): Promise<string> => {
    let url = `${url_prefix}/cells/${cellId}/${dbName}/${drawContour}/${drawScaleBar}/`;
    if (type === "fluo_image" || type === "fluo2_image") {
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
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          const dark = theme.palette.mode === "dark" ? "&dark_mode=true" : "";
          const response = await axios.get(
            `${url_prefix}/cells/${cellId}/${db_name}/replot?degree=${fitDegree}&channel=${channelParam}${dark}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], replot: url },
          }));
          break;
        }
        case "distribution": {
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          const response = await axios.get(
            `${url_prefix}/cells/${db_name}/${cellId}/distribution?channel=${channelParam}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], distribution: url },
          }));
          break;
        }
        case "distribution_normalized": {
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          const response = await axios.get(
            `${url_prefix}/cells/${db_name}/${selectedLabel}/${cellId}/distribution_normalized?channel=${channelParam}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], distribution_normalized: url },
          }));
          break;
        }
        case "path": {
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          setIsLoading(true);
          const response = await axios.get(
            `${url_prefix}/cells/${cellId}/${db_name}/path?degree=${fitDegree}&channel=${channelParam}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], path: url },
          }));
          setIsLoading(false);
          break;
        }
        case "laplacian": {
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          const response = await axios.get(
            `${url_prefix}/cells/${cellId}/${db_name}/laplacian?channel=${channelParam}&brightness_factor=${laplacianBrightness}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], laplacian: url },
          }));
          break;
        }
        case "sobel": {
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          const response = await axios.get(
            `${url_prefix}/cells/${cellId}/${db_name}/sobel?channel=${channelParam}&brightness_factor=${sobelBrightness}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], sobel: url },
          }));
          break;
        }
        case "hu_mask": {
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          const response = await axios.get(
            `${url_prefix}/cells/${cellId}/${db_name}/hu_mask?channel=${channelParam}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], hu_mask: url },
          }));
          break;
        }
        case "map256": {
          const channelParam = map256Source === 'fluo2' ? 2 : 1;
          const imgTypeParam = map256Source === 'ph' ? 'ph' : 'fluo';
          setIsLoading(true);
          const [rawRes, jetRes, clipRes] = await Promise.all([
            axios.get(
              `${url_prefix}/cells/${cellId}/${db_name}/map256?degree=${fitDegree}&channel=${channelParam}&img_type=${imgTypeParam}`,
              { responseType: "blob" }
            ),
            axios.get(
              `${url_prefix}/cells/${cellId}/${db_name}/map256_jet?degree=${fitDegree}&channel=${channelParam}&img_type=${imgTypeParam}`,
              { responseType: "blob" }
            ),
            axios.get(
              `${url_prefix}/cells/${cellId}/${db_name}/map256_clip?degree=${fitDegree}&channel=${channelParam}&img_type=${imgTypeParam}`,
              { responseType: "blob" }
            ),
          ]);
          const rawUrl = URL.createObjectURL(rawRes.data);
          const jetUrl = URL.createObjectURL(jetRes.data);
          const clipUrl = URL.createObjectURL(clipRes.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], map256: rawUrl, map256_jet: jetUrl, map256_clip: clipUrl },
          }));
          setIsLoading(false);
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
          const channelParam = fluoChannel === 'fluo2' ? 2 : 1;
          const response = await axios.get(
            `${url_prefix}/cells/${db_name}/${cellId}/3d?channel=${channelParam}`,
            { responseType: "blob" }
          );
          const url = URL.createObjectURL(response.data);
          setImages((prev) => ({
            ...prev,
            [cellId]: { ...prev[cellId], cloud_points: url },
          }));
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
  }, [cellIds, currentIndex, db_name, drawContour, drawScaleBar, brightnessFactor, hasFluo2]);

  //------------------------------------
  // drawModeが変わったら追加画像をフェッチ
  //------------------------------------
  useEffect(() => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];
    fetchAdditionalImage(drawMode, cellId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawMode, cellIds, currentIndex, fitDegree, fluoChannel, laplacianBrightness, sobelBrightness, map256Source]);

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
  const handleLaplacianBrightnessChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setLaplacianBrightness(parseFloat(e.target.value));
  };
  const handleSobelBrightnessChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setSobelBrightness(parseFloat(e.target.value));
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

  const fallbackCopyToClipboard = (text: string) =>
    new Promise<void>((resolve, reject) => {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.top = "0";
      textarea.style.left = "0";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();

      try {
        const successful = document.execCommand("copy");
        if (!successful) {
          reject(new Error("Fallback copy command was unsuccessful"));
        } else {
          resolve();
        }
      } catch (err) {
        reject(err instanceof Error ? err : new Error("Fallback copy failed"));
      } finally {
        document.body.removeChild(textarea);
      }
    });

  const handleCopyLink = async () => {
    const currentCellId = cellIds[currentIndex];
    if (!currentCellId) {
      setCopyMessage("Cell data is still loading.");
      return;
    }

    try {
      const shareUrl = new URL(window.location.href);
      shareUrl.searchParams.set("db", db_name);
      shareUrl.searchParams.set("db_name", db_name);
      shareUrl.searchParams.set("cell_id", currentCellId);
      shareUrl.searchParams.set("cell", (currentIndex + 1).toString());

      const shareUrlString = shareUrl.toString();

      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrlString);
      } else {
        await fallbackCopyToClipboard(shareUrlString);
      }
      setCopyMessage("Link copied to clipboard!");
    } catch (error) {
      console.error("Failed to copy share link:", error);
      setCopyMessage("Failed to copy link.");
    }
  };

  const handleCopySnackbarClose = (
    _event?: React.SyntheticEvent | Event,
    reason?: string
  ) => {
    if (reason === "clickaway") {
      return;
    }
    setCopyMessage(null);
  };

  // currentIndex が変わったら、テキスト入力も追随しておく
  useEffect(() => {
    setInputIndex((currentIndex + 1).toString());
  }, [currentIndex]);

  //------------------------------------
  // 実際の描画部分
  //------------------------------------
  const isShareDisabled = cellIds.length === 0 || !cellIds[currentIndex];

  return (
    <Box sx={{ p: { xs: 2, md: 3 }, width: "100%" }}>
      {/* パンくずリスト */}
      <Box mb={2} display="flex" alignItems="center" gap={1}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Link underline="hover" color="inherit" href="/dbconsole">
            Database Console
          </Link>
          <Typography color="text.primary">{db_name}</Typography>
        </Breadcrumbs>
        <Tooltip title={isShareDisabled ? "Cell data is still loading" : "Copy share link"}>
          <span>
            <IconButton
              size="small"
              onClick={handleCopyLink}
              aria-label="Copy share link"
              disabled={isShareDisabled}
            >
              <ContentCopyIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
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
                    {LABEL_OPTIONS.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
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
                      sx={{ color: 'text.primary' }}
                    />
                  }
                  label="Contour"
                  sx={{ color: 'text.primary' }}
                />
              </Grid>
              <Grid item xs={2}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={drawScaleBar}
                      onChange={handleScaleBarChange}
                      sx={{ color: 'text.primary' }}
                    />
                  }
                  label="Scale"
                  sx={{ color: 'text.primary' }}
                />
              </Grid>
              <Grid item xs={2}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={autoPlay}
                      onChange={handleAutoPlayChange}
                      sx={{ color: 'text.primary' }}
                    />
                  }
                  label="Auto"
                  sx={{ color: 'text.primary' }}
                />
              </Grid>
              <Grid item xs={12} />
              <Grid item xs={hasFluo2 ? 4 : 6}>
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
              <Grid item xs={hasFluo2 ? 4 : 6}>
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
              {hasFluo2 && (
                <Grid item xs={4}>
                  <FormControl fullWidth variant="outlined">
                    <InputLabel id="fluo-channel-label">Fluo</InputLabel>
                    <Select
                      labelId="fluo-channel-label"
                      label="Fluo"
                      value={fluoChannel}
                      onChange={(e) =>
                        setFluoChannel(e.target.value as 'fluo1' | 'fluo2')
                      }
                    >
                      <MenuItem value="fluo1">fluo1</MenuItem>
                      <MenuItem value="fluo2">fluo2</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              )}
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
              sx={{ backgroundColor: 'primary.main', minWidth: '100px', '&:hover': { backgroundColor: 'primary.dark' } }}
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
              sx={{ backgroundColor: 'primary.main', minWidth: '100px', '&:hover': { backgroundColor: 'primary.dark' } }}
            >
              Next
            </Button>
          </Box>

          {/* PH / Fluo画像 */}
          <Grid container spacing={2}>
            <Grid item xs={12} md={hasFluo2 ? 4 : 6}>
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
            <Grid item xs={12} md={hasFluo2 ? 4 : 6}>
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
            {hasFluo2 && (
              <Grid item xs={12} md={4}>
                {images[cellIds[currentIndex]] && images[cellIds[currentIndex]].fluo2 ? (
                  <img
                    src={images[cellIds[currentIndex]].fluo2 as string}
                    alt={`Cell ${cellIds[currentIndex]} Fluo2`}
                    style={{ width: "100%", height: "auto" }}
                  />
                ) : (
                  <div>Not available</div>
                )}
              </Grid>
            )}
          </Grid>
        </Grid>

        {/* ----- 中央カラム: DrawModeごとの追加表示 ----- */}
        <Grid item xs={12} md={6} lg={4}>
          <Box sx={{ mb: 2 }}>
            {/* DrawMode セレクト + Polyfit Degree */}
            <Box sx={{ display: "flex", gap: 2 }}>
              <FormControl fullWidth variant="outlined" sx={{ flex: 1 }}>
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
              {drawMode === "laplacian" && hasFluo2 && (
                <FormControl variant="outlined" sx={{ minWidth: 120 }}>
                  <InputLabel id="laplacian-channel-label">Fluo</InputLabel>
                  <Select
                    labelId="laplacian-channel-label"
                    label="Fluo"
                    value={fluoChannel}
                    onChange={(e) =>
                      setFluoChannel(e.target.value as "fluo1" | "fluo2")
                    }
                  >
                    <MenuItem value="fluo1">fluo1</MenuItem>
                    <MenuItem value="fluo2">fluo2</MenuItem>
                  </Select>
                </FormControl>
              )}
              {drawMode === "laplacian" && (
                <TextField
                  label="Brightness"
                  variant="outlined"
                  type="number"
                  value={laplacianBrightness}
                  onChange={handleLaplacianBrightnessChange}
                  InputProps={{
                    inputProps: { min: 0.1, step: 0.1 },
                    onWheel: handleWheel,
                    autoComplete: "off",
                  }}
                  sx={{ width: 120 }}
                />
              )}
              {drawMode === "sobel" && hasFluo2 && (
                <FormControl variant="outlined" sx={{ minWidth: 120 }}>
                  <InputLabel id="sobel-channel-label">Fluo</InputLabel>
                  <Select
                    labelId="sobel-channel-label"
                    label="Fluo"
                    value={fluoChannel}
                    onChange={(e) =>
                      setFluoChannel(e.target.value as "fluo1" | "fluo2")
                    }
                  >
                    <MenuItem value="fluo1">fluo1</MenuItem>
                    <MenuItem value="fluo2">fluo2</MenuItem>
                  </Select>
                </FormControl>
              )}
              {drawMode === "sobel" && (
                <TextField
                  label="Brightness"
                  variant="outlined"
                  type="number"
                  value={sobelBrightness}
                  onChange={handleSobelBrightnessChange}
                  InputProps={{
                    inputProps: { min: 0.1, step: 0.1 },
                    onWheel: handleWheel,
                    autoComplete: "off",
                  }}
                  sx={{ width: 120 }}
                />
              )}
              {drawMode === "hu_mask" && hasFluo2 && (
                <FormControl variant="outlined" sx={{ minWidth: 120 }}>
                  <InputLabel id="hu-mask-channel-label">Fluo</InputLabel>
                  <Select
                    labelId="hu-mask-channel-label"
                    label="Fluo"
                    value={fluoChannel}
                    onChange={(e) =>
                      setFluoChannel(e.target.value as "fluo1" | "fluo2")
                    }
                  >
                    <MenuItem value="fluo1">fluo1</MenuItem>
                    <MenuItem value="fluo2">fluo2</MenuItem>
                  </Select>
                </FormControl>
              )}
            </Box>

            {DRAW_MODES.find((m) => m.value === drawMode)?.needsPolyfit && (
              <Box mt={2} sx={{ display: "flex", gap: 2 }}>
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
                  sx={{ flex: 1 }}
                />
                {drawMode === "map256" && (
                  <FormControl variant="outlined" sx={{ minWidth: 120 }}>
                    <InputLabel id="map256-source-label">Channel</InputLabel>
                    <Select
                      labelId="map256-source-label"
                      label="Channel"
                      value={map256Source}
                      onChange={(e) =>
                        setMap256Source(
                          e.target.value as "ph" | "fluo1" | "fluo2"
                        )
                      }
                    >
                      <MenuItem value="ph">ph</MenuItem>
                      <MenuItem value="fluo1">fluo1</MenuItem>
                      {hasFluo2 && <MenuItem value="fluo2">fluo2</MenuItem>}
                    </Select>
                  </FormControl>
                )}
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

            {drawMode === "laplacian" &&
              images[cellIds[currentIndex]]?.laplacian && (
                <img
                  src={images[cellIds[currentIndex]]?.laplacian}
                  alt={`Cell ${cellIds[currentIndex]} Laplacian`}
                  style={{ width: "100%" }}
                />
              )}

            {drawMode === "sobel" &&
              images[cellIds[currentIndex]]?.sobel && (
                <img
                  src={images[cellIds[currentIndex]]?.sobel}
                  alt={`Cell ${cellIds[currentIndex]} Sobel`}
                  style={{ width: "100%" }}
                />
              )}

            {drawMode === "hu_mask" &&
              images[cellIds[currentIndex]]?.hu_mask && (
                <img
                  src={images[cellIds[currentIndex]]?.hu_mask}
                  alt={`Cell ${cellIds[currentIndex]} HU Mask`}
                  style={{ width: "100%" }}
                />
              )}

            {drawMode === "map256" && isLoading ? (
              <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                style={{ height: 400 }}
              >
                <Spinner />
              </Box>
            ) : (
              drawMode === "map256" &&
              images[cellIds[currentIndex]]?.map256 && (
                <Box>
                  <img
                    src={images[cellIds[currentIndex]]?.map256}
                    alt={`Cell ${cellIds[currentIndex]} Map256`}
                    style={{ width: "100%" }}
                  />
                  {images[cellIds[currentIndex]]?.map256_clip && (
                    <img
                      src={images[cellIds[currentIndex]]?.map256_clip}
                      alt={`Cell ${cellIds[currentIndex]} Map256 Clip`}
                      style={{ width: "100%" }}
                    />
                  )}
                  {images[cellIds[currentIndex]]?.map256_jet && (
                    <img
                      src={images[cellIds[currentIndex]]?.map256_jet}
                      alt={`Cell ${cellIds[currentIndex]} Map256 Jet`}
                      style={{ width: "100%" }}
                    />
                  )}
                </Box>
              )
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

          {(engineMode === "SDEngine" ||
            engineMode === "PixelCVEngine" ||
            engineMode === "PixelEngine") && (
            <Box sx={{ mb: 2 }}>
              <FormControl fullWidth variant="outlined">
                <InputLabel id="stat-source-label">Channel</InputLabel>
                <Select
                  labelId="stat-source-label"
                  label="Channel"
                  value={statSource}
                  onChange={(e) =>
                    setStatSource(e.target.value as "ph" | "fluo1" | "fluo2")
                  }
                >
                  <MenuItem value="ph">ph</MenuItem>
                  <MenuItem value="fluo1">fluo1</MenuItem>
                  {hasFluo2 && <MenuItem value="fluo2">fluo2</MenuItem>}
                </Select>
              </FormControl>
            </Box>
          )}

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

          {engineMode === "IbpAEngine" && (
            <Box mt={6}>
              <IbpAEngine
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

          {engineMode === "SDEngine" && (
            <Box mt={6}>
              <SDEngine
                dbName={db_name}
                label={selectedLabel}
                cellId={cellIds[currentIndex]}
                imgType={statSource}
              />
            </Box>
          )}

          {engineMode === "PixelCVEngine" && (
            <Box mt={6}>
              <PixelCVEngine
                dbName={db_name}
                label={selectedLabel}
                cellId={cellIds[currentIndex]}
                imgType={statSource}
              />
            </Box>
          )}

          {engineMode === "PixelEngine" && (
            <Box mt={6}>
              <PixelEngine
                dbName={db_name}
                label={selectedLabel}
                imgType={statSource}
                labelOptions={LABEL_OPTIONS}
                onLabelChange={(value) => setSelectedLabel(value)}
              />
            </Box>
          )}

          {engineMode === "AreaFractionEngine" && (
            <Box mt={6}>
              <AreaFractionEngine
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
      <Snackbar
        open={Boolean(copyMessage)}
        autoHideDuration={3000}
        onClose={handleCopySnackbarClose}
        message={copyMessage ?? ""}
      />
    </Box>
  );
};

export default CellImageGrid;
