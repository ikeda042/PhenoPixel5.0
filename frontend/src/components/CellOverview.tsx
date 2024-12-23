import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import {
  Stack,
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
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

<<<<<<< HEAD
/**
 * 画像情報を保持する型
 */
type ImageState = {
    ph: string;                 // 位相差画像 (必須)
    fluo?: string | null;       // 蛍光画像 (DBによってはない)
    replot?: string;            // プロット再描画時の画像
    distribution?: string;      // 分布画像
    path?: string;              // ピークパス画像
    prediction?: string;        // 推論画像
    cloud_points?: string;      // 3Dフルオロ画像
    cloud_points_ph?: string;   // 3D位相差画像
};

=======
//-----------------------------------
// 型定義
//-----------------------------------
type ImageState = {
  ph: string;               // 位相差画像URL
  fluo?: string | null;     // 蛍光画像URL
  replot?: string;          // 再プロット画像URL
  distribution?: string;    // 分布図画像URL (非正規化)
  distribution_normalized?: string; // 分布図画像URL (正規化)
  path?: string;            // Peak-path画像URL
  prediction?: string;      // T1推定画像URL
  cloud_points?: string;    // 3D蛍光点群画像URL
  cloud_points_ph?: string; // 3D位相差点群画像URL
};

// 「どのモードにするか」を列挙型的に管理する
type DrawModeType =
  | "light"
  | "replot"
  | "distribution"
  | "distribution_normalized" // ★ 追加
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
// まとめて管理したい設定たち
//-----------------------------------
>>>>>>> main
const url_prefix = settings.url_prefix;

// DrawModeごとに、表示名や「Polyfit Degree入力が必要かどうか」をまとめた設定
const DRAW_MODES: {
  value: DrawModeType;
  label: string;              // セレクトボックスで表示するラベル
  needsPolyfit?: boolean;     // Polyfit Degreeを入力させたい場合はtrue
}[] = [
  { value: "light", label: "Light" },
  { value: "replot", label: "Replot", needsPolyfit: true },
  { value: "distribution", label: "Distribution" },
  // ★ Normalized 分布のモードを追加
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
<<<<<<< HEAD
    // クエリパラメータからデフォルト値を取得
    const [searchParams] = useSearchParams();
    const db_name = searchParams.get('db_name') ?? "test_database.db";
    const cell_number = searchParams.get('cell') ?? "1";
    const init_draw_mode = searchParams.get('init_draw_mode') ?? "light";

    // 各種 state
    const [cellIds, setCellIds] = useState<string[]>([]);
    const [images, setImages] = useState<{ [key: string]: ImageState }>({});
    const [selectedLabel, setSelectedLabel] = useState<string>("74");
    const [manualLabel, setManualLabel] = useState<string>("");
    const [currentIndex, setCurrentIndex] = useState<number>(parseInt(cell_number) - 1);
    const [drawContour, setDrawContour] = useState<boolean>(true);
    const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
    const [autoPlay, setAutoPlay] = useState<boolean>(false);
    const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
    const [contourData, setContourData] = useState<number[][]>([]);
    const [contourDataT1, setContourDataT1] = useState<number[][]>([]);
    const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null);
    const [drawMode, setDrawMode] = useState<string>(init_draw_mode);
    const [fitDegree, setFitDegree] = useState<number>(4);
    const [isLoading, setIsLoading] = useState(false);
    const [engineMode, setEngineMode] = useState<string>("None");

    // デバウンス用
    const lastCallTimeRef = useRef<number | null>(null);
    const debounce = (func: () => void, wait: number) => {
        return () => {
            const now = new Date().getTime();
            if (lastCallTimeRef.current === null || (now - lastCallTimeRef.current) > wait) {
                lastCallTimeRef.current = now;
                func();
            }
        };
=======
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

  // 各種スイッチ・入力
  const [drawContour, setDrawContour] = useState<boolean>(true);
  const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
  const [autoPlay, setAutoPlay] = useState<boolean>(false);
  const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
  const [drawMode, setDrawMode] = useState<DrawModeType>(init_draw_mode);
  const [fitDegree, setFitDegree] = useState<number>(4);
  const [engineMode, setEngineMode] = useState<EngineName>("None");

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
>>>>>>> main
    };
  };

<<<<<<< HEAD
    /**
     * DBからセルID一覧を取得
     */
    useEffect(() => {
        const fetchCellIds = async () => {
            const response = await axios.get(`${url_prefix}/cells/${db_name}/${selectedLabel}`);
            const ids = response.data.map((cell: { cell_id: string }) => cell.cell_id);
            setCellIds(ids);
        };
        fetchCellIds();
    }, [db_name, selectedLabel]);

    /**
     * セルに紐づく PH画像・Fluo画像を取得
     */
    useEffect(() => {
        const fetchImages = async (cellId: string) => {
            try {
                // PH画像 or Fluo画像を共通化して取得する内部関数
                const fetchImage = async (
                    type: 'ph_image' | 'fluo_image', 
                    brightnessFactor: number = 1.0
                ) => {
                    let url = `${url_prefix}/cells/${cellId}/${db_name}/${drawContour}/${drawScaleBar}/`;
                    if (type === 'fluo_image') {
                        url += `${type}?brightness_factor=${brightnessFactor}`;
                    } else {
                        url += `${type}`;
                    }
                    console.log(`Fetching image from URL: ${url}`);
                    const response = await axios.get(url, { responseType: 'blob' });
                    const imageUrl = URL.createObjectURL(response.data);

                    // 画像サイズを記憶
                    const imageDimensions = await new Promise<{ width: number; height: number }>((resolve, reject) => {
                        const img = new Image();
                        img.onload = () => resolve({ width: img.width, height: img.height });
                        img.onerror = reject;
                        img.src = imageUrl;
                    });
                    setImageDimensions(imageDimensions);
                    return imageUrl;
                };

                // PH画像をまず取得
                const phImage = await fetchImage('ph_image');
                // single_layer でなければ Fluo画像を取得
                let fluoImage: string | null = null;
                if (!db_name.includes("single_layer")) {
                    fluoImage = await fetchImage('fluo_image', brightnessFactor);
                }

                return { ph: phImage, fluo: fluoImage };
            } catch (error) {
                console.error("Error fetching images: FE", error);
                return null;
            }
        };

        // 輪郭情報も合わせて取得
        const handleFetchImages = async (cellId: string) => {
            const newImages = await fetchImages(cellId);
            if (newImages) {
                setImages((prevImages) => ({
                    ...prevImages,
                    [cellId]: { ...prevImages[cellId], ...newImages }
                }));
                fetchContour(cellId);
                fetchContourT1(cellId);
            }
        };

        // cellIds が取得できている場合のみ実行
        if (cellIds.length > 0) {
            handleFetchImages(cellIds[currentIndex]);
        }
    }, [cellIds, currentIndex, db_name, drawContour, drawScaleBar, brightnessFactor]);

    /**
     * 初期のラベルを取得して manualLabel に反映
     */
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

    /**
     * 輪郭を取得
     */
    const fetchContour = async (cellId: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cells/${cellId}/contour/raw?db_name=${db_name}`);
            setContourData(response.data.contour);
        } catch (error) {
            console.error("Error fetching contour data:", error);
        }
=======
  //------------------------------------
  // セル一覧の取得
  //------------------------------------
  useEffect(() => {
    const fetchCellIds = async () => {
      const response = await axios.get(`${url_prefix}/cells/${db_name}/${selectedLabel}`);
      const ids = response.data.map((cell: { cell_id: string }) => cell.cell_id);
      setCellIds(ids);
>>>>>>> main
    };
    fetchCellIds();
  }, [db_name, selectedLabel]);

<<<<<<< HEAD
    /**
     * Torch推論用の輪郭(モデルT1)を取得
     */
    const fetchContourT1 = async (cellId: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cell_ai/${db_name}/${cellId}/plot_data`);
            setContourDataT1(response.data);
        } catch (error) {
            console.error("Error fetching contour data:", error);
        }
    };

    /**
     * 再プロット画像を取得
     */
    const fetchReplotImage = async (cellId: string, dbName: string, fitDegree: number) => {
        try {
            const response = await axios.get(
                `${url_prefix}/cells/${cellId}/${dbName}/replot?degree=${fitDegree}`, 
                { responseType: 'blob' }
            );
            const replotImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], replot: replotImageUrl }
            }));
        } catch (error) {
            console.error("Error fetching replot image:", error);
        }
    };

    /**
     * ピークパスを取得
     */
    const fetchPeakPath = async (cellId: string, dbName: string, fitDegree: number) => {
        setIsLoading(true);
        try {
            const response = await axios.get(
                `${url_prefix}/cells/${cellId}/${dbName}/path?degree=${fitDegree}`,
                { responseType: 'blob' }
            );
            const pathImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], path: pathImageUrl }
            }));
        } catch (error) {
            console.error("Error fetching peak path:", error);
        } finally {
            setIsLoading(false);
        }
    };

    /**
     * drawMode が replot に切り替わったら再プロット画像を読み込み
     */
    useEffect(() => {
        if (drawMode === "replot" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.replot) {
                fetchReplotImage(cellId, db_name, fitDegree);
            }
        }
    }, [drawMode, cellIds, currentIndex, fitDegree]);

    /**
     * drawMode が distribution に切り替わったら分布画像を読み込み
     */
    useEffect(() => {
        if (drawMode === "distribution" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.distribution) {
                fetchDistributionImage(cellId, db_name, selectedLabel);
            }
        }
    }, [drawMode, cellIds, currentIndex, selectedLabel]);

    /**
     * drawMode が path に切り替わったらピークパス画像を読み込み
     */
    useEffect(() => {
        if (drawMode === "path" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.path) {
                fetchPeakPath(cellId, db_name, fitDegree);
            }
        }
    }, [drawMode, cellIds, currentIndex, fitDegree]);

    /**
     * Nextボタン (デバウンス付き)
     */
    const handleNext = debounce(() => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % cellIds.length);
    }, 500);

    /**
     * Prevボタン
     */
    const handlePrev = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + cellIds.length) % cellIds.length);
    };

    /**
     * ラベルリスト(「All」「1」など) が変更されたとき
     */
    const handleLabelChange = (event: SelectChangeEvent<string>) => {
        setSelectedLabel(event.target.value);
    };

    /**
     * Manual Label (セル固有のラベル) が変更されたとき
     */
    const handleCellLabelChange = async (event: SelectChangeEvent<string>) => {
        const newLabel = event.target.value;
        const cellId = cellIds[currentIndex];

        try {
            await axios.patch(`${url_prefix}/cells/${db_name}/${cellId}/${newLabel}`);
            setManualLabel(newLabel);
        } catch (error) {
            console.error("Error updating cell label:", error);
        }
    };

    /**
     * チェックボックス(Contour表示)の変更
     */
    const handleContourChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setDrawContour(e.target.checked);
    };

    /**
     * チェックボックス(スケールバー表示)の変更
     */
    const handleScaleBarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setDrawScaleBar(e.target.checked);
    };

    /**
     * チェックボックス(AutoPlay)の変更
     */
    const handleAutoPlayChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setAutoPlay(e.target.checked);
    };

    /**
     * 明るさ係数をテキストフィールドで入力
     */
    const handleBrightnessChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setBrightnessFactor(parseFloat(e.target.value));
    };

    /**
     * Number入力欄でスクロールしても数値が変更されないようにする
     */
    const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
        event.currentTarget.blur();
        event.preventDefault();
    };

    /**
     * drawMode(描画の種類) の変更
     */
    const handleDrawModeChange = (event: SelectChangeEvent<string>) => {
        setDrawMode(event.target.value);
    };

    /**
     * ポリフィットの次数（degree）
     */
    const handleFitDegreeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFitDegree(parseInt(e.target.value));
    };

    /**
     * キーボードイベント(EnterでNext, '1','2','3','n'でラベル変更等)
     */
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Enter') {
                handleNext();  // EnterキーでNext
            } else if (['1', '2', '3', 'n'].includes(event.key)) {
                let newLabel = event.key;
                if (newLabel === 'n') {
                    newLabel = "1000";  // 'n' は "1000" に対応
                }
                setManualLabel(newLabel);
                handleCellLabelChange({ target: { value: newLabel } } as SelectChangeEvent<string>);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [cellIds, currentIndex]);

    /**
     * MorphoEngine 系モードの変更
     */
    const handleEngineModeChange = (event: SelectChangeEvent<string>) => {
        setEngineMode(event.target.value);
    };

    type EngineName = 'None' | 'MorphoEngine 2.0' | 'MedianEngine' | 'MeanEngine' | 'HeatmapEngine' | 'VarEngine';

    const engineLogos: Record<EngineName, string> = {
        None: 'path_to_none_logo.png',
        'MorphoEngine 2.0': '/logo_tp.png',
        'MedianEngine': '/logo_dots.png',
        'MeanEngine': '/logo_circular.png',
        'VarEngine': '/var_logo.png',
        'HeatmapEngine': '/logo_heatmap.png',
    };

    /**
     * AutoPlayがオンの場合、3秒ごとにNext
     */
    useEffect(() => {
        let autoNextInterval: NodeJS.Timeout | undefined;
        if (autoPlay) {
            autoNextInterval = setInterval(handleNext, 3000);  // 3秒ごとにNext
        }
        return () => {
            if (autoNextInterval) clearInterval(autoNextInterval);
        };
    }, [autoPlay]);

    /**
     * 輪郭描画用データ
     */
    const contourPlotData = {
        datasets: [
            {
                label: 'Contour',
                data: contourData.map(point => ({ x: point[0], y: point[1] })),
                borderColor: 'lime',
                backgroundColor: 'lime',
                pointRadius: 1,
            },
            ...(drawMode === "t1contour"
                ? [{
                    label: 'Model T1',
                    data: contourDataT1.map(point => ({ x: point[0], y: point[1] })),
                    borderColor: 'red',
                    backgroundColor: 'red',
                    pointRadius: 1,
                }] 
                : []
            )
        ]
    };

    /**
     * 推論結果の画像を取得
     */
    const fetchPredictionImage = async (cellId: string, dbName: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cell_ai/${dbName}/${cellId}`, { responseType: 'blob' });
            const predictionImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], prediction: predictionImageUrl }
            }));
        } catch (error) {
            console.error("Error fetching prediction image:", error);
        }
    };

    /**
     * 分布画像を取得
     */
    const fetchDistributionImage = async (cellId: string, dbName: string, label: string) => {
        try {
            const response = await axios.get(
                `${url_prefix}/cells/${dbName}/${cellId}/distribution`, 
                { responseType: 'blob' }
            );
            const distributionImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { 
                    ...prevImages[cellId], 
                    distribution: distributionImageUrl 
                }
            }));
        } catch (error) {
            console.error("Error fetching distribution image:", error);
        }
    };

    /**
     * drawMode が prediction に切り替わったら推論画像を取得
     */
    useEffect(() => {
        if (drawMode === "prediction" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.prediction) {
                fetchPredictionImage(cellId, db_name);
            }
        }
    }, [drawMode, cellIds, currentIndex]);

    /**
     * 3D Fluo画像を取得
     */
    const fetch3DImage = async (cellId: string, dbName: string) => {
        try {
            const response = await axios.get(
                `${url_prefix}/cells/${dbName}/${cellId}/3d`, 
                { responseType: 'blob' }
            );
            const imageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], cloud_points: imageUrl }
            }));
        } catch (error) {
            console.error("Error fetching 3D image:", error);
        }
    };

    /**
     * drawMode が cloud_points に切り替わったら3D Fluo画像を取得
     */
    useEffect(() => {
        if (drawMode === "cloud_points" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.cloud_points) {
                fetch3DImage(cellId, db_name);
            }
        }
    }, [drawMode, cellIds, currentIndex]);

    /**
     * 3D PH画像を取得
     */
    const fetch3DPhImage = async (cellId: string, dbName: string) => {
        try {
            const response = await axios.get(
                `${url_prefix}/cells/${dbName}/${cellId}/3d-ph`, 
                { responseType: 'blob' }
            );
            const imageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], cloud_points_ph: imageUrl }
            }));
        } catch (error) {
            console.error("Error fetching 3D image:", error);
        }
    };

    /**
     * drawMode が cloud_points_ph に切り替わったら3D PH画像を取得
     */
    useEffect(() => {
        if (drawMode === "cloud_points_ph" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.cloud_points_ph) {
                fetch3DPhImage(cellId, db_name);
            }
        }
    }, [drawMode, cellIds, currentIndex]);

    /**
     * scatter 用オプション
     */
    const contourPlotOptions: ChartOptions<'scatter'> = {
        maintainAspectRatio: true,
        aspectRatio: 1,
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                min: 0,
                max: imageDimensions?.width,
                reverse: false,
            },
            y: {
                type: 'linear',
                min: 0,
                max: imageDimensions?.height,
                reverse: true
            }
=======
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
    // ※ 「PH画像」「蛍光画像」など、ほぼ常に表示する可能性があるものはここでまとめて取得
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
>>>>>>> main
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
  // （replot, path, distribution, prediction, cloud_pointsなど）
  //------------------------------------
  const fetchAdditionalImage = async (mode: DrawModeType, cellId: string) => {
    try {
      switch (mode) {
        //----------------------------------------
        // Replot
        //----------------------------------------
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

        //----------------------------------------
        // Distribution (非正規化)
        //----------------------------------------
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

        //----------------------------------------
        // Distribution (正規化) ★ 追加分 ★
        //----------------------------------------
        case "distribution_normalized": {
          if (!images[cellId]?.distribution_normalized) {
            // labelは selectedLabel でもよいが、
            // APIパスに書かれている {label} は空文字以外のラベルを想定しているなら、
            // 例えば "74" 等を使う or " " を指定するなど、要件に応じて切り替えてください。
            // ここではひとまず selectedLabel を使用してみます。
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

        //----------------------------------------
        // Path
        //----------------------------------------
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

        //----------------------------------------
        // Torch GPU Prediction
        //----------------------------------------
        case "prediction": {
          if (!images[cellId]?.prediction) {
            const response = await axios.get(
              `${url_prefix}/cell_ai/${db_name}/${cellId}`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            setImages((prev) => ({
              ...prev,
              [cellId]: { ...prev[cellId], prediction: url },
            }));
          }
          break;
        }

        //----------------------------------------
        // 3D Fluo
        //----------------------------------------
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

        //----------------------------------------
        // 3D PH
        //----------------------------------------
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

        //----------------------------------------
        // T1 Contour (Scatter で描画するので画像不要)
        //----------------------------------------
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
  // （PH画像、Fluo画像、輪郭データなどを取得）
  //------------------------------------
  useEffect(() => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];
    // まずは基本画像(PH,Fluo)や輪郭をまとめて取得
    fetchStandardImages(cellId);
    fetchContour(cellId);
    fetchContourT1(cellId);
  }, [cellIds, currentIndex, db_name, drawContour, drawScaleBar, brightnessFactor]);

  //------------------------------------
  // drawModeが変わったら追加画像をフェッチ
  //------------------------------------
  useEffect(() => {
    if (cellIds.length === 0) return;
    const cellId = cellIds[currentIndex];
    fetchAdditionalImage(drawMode, cellId);
  }, [drawMode, cellIds, currentIndex, fitDegree]);

  //------------------------------------
  // 現在のセルIDに対応した初期ラベルを取得
  // （ページ切り替えやロード時の一度きり）
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
<<<<<<< HEAD

    /**
     * drawMode 切り替えによる表示内容を一括管理する
     */
    const renderDrawModeContent = () => {
        // 画像やチャートなど、drawMode に応じて表示を切り替える
        switch (drawMode) {
            case "light":
                // PH輪郭 (T1は表示しない)
                return <Scatter data={contourPlotData} options={contourPlotOptions} />;

            case "replot":
                return images[cellIds[currentIndex]]?.replot ? (
                    <img
                        src={images[cellIds[currentIndex]]?.replot}
                        alt={`Cell ${cellIds[currentIndex]} Replot`}
                        style={{ width: "100%" }}
                    />
                ) : (
                    <div>Loading Replot...</div>
                );

            case "distribution":
                return images[cellIds[currentIndex]]?.distribution ? (
                    <img
                        src={images[cellIds[currentIndex]]?.distribution}
                        alt={`Cell ${cellIds[currentIndex]} Distribution`}
                        style={{ width: "100%" }}
                    />
                ) : (
                    <div>Loading Distribution...</div>
                );

            case "path":
                if (isLoading) {
                    return (
                        <Box
                            display="flex"
                            justifyContent="center"
                            alignItems="center"
                            style={{ height: 400 }}
                        >
                            <Spinner />
                        </Box>
                    );
                } else {
                    return images[cellIds[currentIndex]]?.path ? (
                        <img
                            src={images[cellIds[currentIndex]]?.path}
                            alt={`Cell ${cellIds[currentIndex]} Path`}
                            style={{ width: "100%" }}
                        />
                    ) : (
                        <div>Loading Peak-path...</div>
                    );
                }

            case "prediction":
                return images[cellIds[currentIndex]]?.prediction ? (
                    <img
                        src={images[cellIds[currentIndex]]?.prediction}
                        alt={`Cell ${cellIds[currentIndex]} Prediction`}
                        style={{ width: "100%" }}
                    />
                ) : (
                    <div>Loading Prediction...</div>
                );

            case "t1contour":
                // PH輪郭 + T1輪郭 両方
                return (
                    <Scatter
                        data={contourPlotData}
                        options={contourPlotOptions}
                    />
                );

            case "cloud_points":
                return images[cellIds[currentIndex]]?.cloud_points ? (
                    <img
                        src={images[cellIds[currentIndex]]?.cloud_points}
                        alt={`Cell ${cellIds[currentIndex]} 3D Fluo`}
                        style={{ width: "100%" }}
                    />
                ) : (
                    <div>Loading 3D Fluo...</div>
                );

            case "cloud_points_ph":
                return images[cellIds[currentIndex]]?.cloud_points_ph ? (
                    <img
                        src={images[cellIds[currentIndex]]?.cloud_points_ph}
                        alt={`Cell ${cellIds[currentIndex]} 3D PH`}
                        style={{ width: "100%" }}
                    />
                ) : (
                    <div>Loading 3D PH...</div>
                );

            default:
                return <div>No draw mode selected</div>;
        }
    };

    // 「replot」「path」のいずれかの場合はPolyFitDegreeが必要
    const isPolyFitMode = ["replot", "path"].includes(drawMode);

    return (
        <>
            {/* パンくずリスト */}
            <Box>
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

            <Stack direction="row" spacing={2} sx={{ marginTop: 8 }}>
                {/* 左ペイン: セル一覧・ラベル・PH/Fluo画像 */}
                <Box sx={{ width: 580, height: 420, marginLeft: 2 }}>
                    {/* セレクト: 大きめのラベル (All / N/A / 1,2,3など) */}
                    <FormControl fullWidth>
                        <InputLabel id="label-select-label">Label</InputLabel>
                        <Select
                            labelId="label-select-label"
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

                    {/* オプションチェック類 */}
                    <Box mt={2}>
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
                                    label="Brightness Factor"
                                    type="number"
                                    value={brightnessFactor}
                                    onChange={handleBrightnessChange}
                                    InputProps={{
                                        inputProps: { min: 0.1, step: 0.1 },
                                        onWheel: handleWheel,
                                        autoComplete: "off"
                                    }}
                                />
                            </Grid>
                            <Grid item xs={3}>
                                <FormControl fullWidth>
                                    <InputLabel id="manual-label-select-label">Manual Label</InputLabel>
                                    <Select
                                        labelId="manual-label-select-label"
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

                    {/* Prev / Next ボタンなど */}
                    <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={handlePrev}
                            disabled={cellIds.length === 0}
                            style={{ backgroundColor: "black", minWidth: "100px" }}
                        >
                            Prev
                        </Button>
                        <Typography variant="h6">
                            {cellIds.length > 0
                                ? `Cell ${currentIndex + 1} of ${cellIds.length}`
                                : `Cell ${currentIndex} of ${cellIds.length}`}{" "}
                            / ({cellIds[currentIndex]})
                        </Typography>
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

                    {/* PH/Fluo画像表示 */}
                    <Grid container spacing={2} style={{ marginTop: 20 }}>
                        <Grid item xs={6}>
                            {images[cellIds[currentIndex]] ? (
                                <img
                                    src={images[cellIds[currentIndex]].ph}
                                    alt={`Cell ${cellIds[currentIndex]} PH`}
                                    style={{ width: "100%" }}
                                />
                            ) : (
                                <div>Loading PH...</div>
                            )}
                        </Grid>
                        <Grid item xs={6}>
                            {images[cellIds[currentIndex]] && images[cellIds[currentIndex]].fluo ? (
                                <img
                                    src={images[cellIds[currentIndex]].fluo as string}
                                    alt={`Cell ${cellIds[currentIndex]} Fluo`}
                                    style={{ width: "100%" }}
                                />
                            ) : db_name.includes("single_layer") ? (
                                <Box
                                    sx={{
                                        display: 'flex',
                                        justifyContent: 'center',
                                        alignItems: 'center',
                                        height: '100%'
                                    }}
                                >
                                    <Typography variant="h5">Single layer mode.</Typography>
                                    <img
                                        src="/logo_dots.png"
                                        alt="Morpho Engine is off"
                                        style={{ maxWidth: '15%', maxHeight: '15%' }}
                                    />
                                </Box>
                            ) : (
                                <div>Not available</div>
                            )}
                        </Grid>
                    </Grid>
                </Box>

                {/* 中央ペイン: drawMode の共通セクション */}
                <Box sx={{ width: 420, height: 420, marginLeft: 2 }}>
                    {/* ◆◆◆ ここが drawMode 選択をまとめたブロック ◆◆◆ */}
                    <Grid container spacing={2}>
                        {isPolyFitMode ? (
                            // replot や path の場合: 8:4 で分割
                            <>
                                <Grid item xs={8}>
                                    <FormControl fullWidth>
                                        <InputLabel id="draw-mode-select-label">Draw Mode</InputLabel>
                                        <Select
                                            labelId="draw-mode-select-label"
                                            value={drawMode}
                                            onChange={handleDrawModeChange}
                                        >
                                            <MenuItem value="light">Light</MenuItem>
                                            <MenuItem value="replot">Replot</MenuItem>
                                            <MenuItem value="distribution">Distribution</MenuItem>
                                            <MenuItem value="path">Peak-path</MenuItem>
                                            <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                            <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                            <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                            <MenuItem value="cloud_points_ph">3D PH</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                                <Grid item xs={4}>
                                    <TextField
                                        label="Polyfit Degree"
                                        type="number"
                                        value={fitDegree}
                                        onChange={handleFitDegreeChange}
                                        InputProps={{
                                            inputProps: { min: 0, step: 1 },
                                            onWheel: handleWheel,
                                            autoComplete: "off"
                                        }}
                                    />
                                </Grid>
                            </>
                        ) : (
                            // それ以外のモード: 横幅いっぱい (xs={12}) でプルダウンを表示
                            <Grid item xs={12}>
                                <FormControl fullWidth>
                                    <InputLabel id="draw-mode-select-label">Draw Mode</InputLabel>
                                    <Select
                                        labelId="draw-mode-select-label"
                                        value={drawMode}
                                        onChange={handleDrawModeChange}
                                    >
                                        <MenuItem value="light">Light</MenuItem>
                                        <MenuItem value="replot">Replot</MenuItem>
                                        <MenuItem value="distribution">Distribution</MenuItem>
                                        <MenuItem value="path">Peak-path</MenuItem>
                                        <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                        <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                        <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                        <MenuItem value="cloud_points_ph">3D PH</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                        )}
                    </Grid>

                    {/* 実際に選んだ drawMode に応じた内容を表示する */}
                    <Box mt={2}>
                        {renderDrawModeContent()}
                    </Box>
                </Box>

                {/* 右ペイン: MorphoEngine テーブル or 各Engineコンポーネント */}
                <Box sx={{ width: 350, height: 420, marginLeft: 2 }}>
                    <FormControl fullWidth>
                        <InputLabel id="engine-select-label">MorphoEngine</InputLabel>
                        <Select
                            labelId="engine-select-label"
                            id="engine-select"
                            value={engineMode}
                            onChange={handleEngineModeChange}
                            renderValue={(selected: string) => {
                                if (selected === 'None') {
                                    return "None";
                                } else {
                                    const engineName = selected as EngineName;
                                    let displayText: string = engineName;
                                    if (engineName === 'MorphoEngine 2.0') {
                                        displayText = engineName;
                                    } else if (engineName === 'MedianEngine') {
                                        displayText = "MedianEngine";
                                    } else if (engineName === 'MeanEngine') {
                                        displayText = "MeanEngine";
                                    }
                                    return (
                                        <Box display="flex" alignItems="center">
                                            {engineName !== 'None' && (
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
                                        {engine !== 'None' && (
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

                    {/* EngineMode ごとの表示内容 */}
                    {engineMode === "None" && (
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center',
                                height: '100%'
                            }}
                        >
                            <Typography variant="h5">MorphoEngine is off.</Typography>
                            <img
                                src="/logo_tp.png"
                                alt="Morpho Engine is off"
                                style={{ maxWidth: '15%', maxHeight: '15%' }}
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
                </Box>
            </Stack>
        </>
    );
=======
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

  // AutoPlay
  useEffect(() => {
    let autoNextInterval: NodeJS.Timeout | undefined;
    if (autoPlay) {
      autoNextInterval = setInterval(handleNext, 3000); // 3秒ごとに自動でNext
    }
    return () => {
      if (autoNextInterval) clearInterval(autoNextInterval);
    };
  }, [autoPlay]);

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
    // 数値入力でスクロールが効いてしまうのを抑制
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
  // キーボードイベント
  // Enterキー: Next
  // 1,2,3,n キー: ラベル変更
  //------------------------------------
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Enter") {
        handleNext(); // EnterキーでNextを押す
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
  // 実際の描画部分
  //------------------------------------
  return (
    <>
      {/* パンくずリスト */}
      <Box>
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

      <Stack direction="row" spacing={2} sx={{ marginTop: 8 }}>
        {/* 左カラム: PH/Fluoや、Prev/Nextなど */}
        <Box sx={{ width: 580, height: 420, marginLeft: 2 }}>
          {/* ラベル選択 */}
          <FormControl fullWidth>
            <InputLabel id="label-select-label">Label</InputLabel>
            <Select labelId="label-select-label" value={selectedLabel} onChange={handleLabelChange}>
              <MenuItem value="74">All</MenuItem>
              <MenuItem value="1000">N/A</MenuItem>
              <MenuItem value="1">1</MenuItem>
              <MenuItem value="2">2</MenuItem>
              <MenuItem value="3">3</MenuItem>
            </Select>
          </FormControl>

          {/* チェックボックス類 */}
          <Box mt={2}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={2}>
                <FormControlLabel
                  control={<Checkbox checked={drawContour} onChange={handleContourChange} style={{ color: "black" }} />}
                  label="Contour"
                  style={{ color: "black" }}
                />
              </Grid>
              <Grid item xs={2}>
                <FormControlLabel
                  control={<Checkbox checked={drawScaleBar} onChange={handleScaleBarChange} style={{ color: "black" }} />}
                  label="Scale"
                  style={{ color: "black" }}
                />
              </Grid>
              <Grid item xs={2}>
                <FormControlLabel
                  control={<Checkbox checked={autoPlay} onChange={handleAutoPlayChange} style={{ color: "black" }} />}
                  label="Auto"
                  style={{ color: "black" }}
                />
              </Grid>
              <Grid item xs={3}>
                <TextField
                  label="Brightness Factor"
                  type="number"
                  value={brightnessFactor}
                  onChange={handleBrightnessChange}
                  InputProps={{
                    inputProps: { min: 0.1, step: 0.1 },
                    onWheel: handleWheel,
                    autoComplete: "off",
                  }}
                />
              </Grid>
              <Grid item xs={3}>
                <FormControl fullWidth>
                  <InputLabel id="manual-label-select-label">Manual Label</InputLabel>
                  <Select labelId="manual-label-select-label" value={manualLabel} onChange={handleCellLabelChange}>
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
          <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
            <Button
              variant="contained"
              color="primary"
              onClick={handlePrev}
              disabled={cellIds.length === 0}
              style={{ backgroundColor: "black", minWidth: "100px" }}
            >
              Prev
            </Button>
            <Typography variant="h6">
              {cellIds.length > 0
                ? `Cell ${currentIndex + 1} of ${cellIds.length}`
                : `Cell ${currentIndex} of ${cellIds.length}`}{" "}
              / ({cellIds[currentIndex]})
            </Typography>
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
          <Grid container spacing={2} style={{ marginTop: 20 }}>
            <Grid item xs={6}>
              {images[cellIds[currentIndex]] ? (
                <img
                  src={images[cellIds[currentIndex]].ph}
                  alt={`Cell ${cellIds[currentIndex]} PH`}
                  style={{ width: "100%" }}
                />
              ) : (
                <div>Loading PH...</div>
              )}
            </Grid>
            <Grid item xs={6}>
              {images[cellIds[currentIndex]] && images[cellIds[currentIndex]].fluo ? (
                <img
                  src={images[cellIds[currentIndex]].fluo as string}
                  alt={`Cell ${cellIds[currentIndex]} Fluo`}
                  style={{ width: "100%" }}
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
        </Box>

        {/* 中央カラム: DrawModeごとの追加表示 */}
        <Box sx={{ width: 420, height: 420, marginLeft: 2 }}>
          {/* DrawMode セレクトと、モードによって必要な入力コンポーネント */}
          <FormControl fullWidth>
            <InputLabel id="draw-mode-select-label">Draw Mode</InputLabel>
            <Select labelId="draw-mode-select-label" value={drawMode} onChange={handleDrawModeChange}>
              {DRAW_MODES.map((mode) => (
                <MenuItem key={mode.value} value={mode.value}>
                  {mode.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Polyfit Degree が必要なモードの場合のみ表示 */}
          {DRAW_MODES.find((m) => m.value === drawMode)?.needsPolyfit && (
            <Box mt={2}>
              <TextField
                label="Polyfit Degree"
                type="number"
                value={fitDegree}
                onChange={handleFitDegreeChange}
                InputProps={{
                  inputProps: { min: 0, step: 1 },
                  onWheel: handleWheel,
                  autoComplete: "off",
                }}
              />
            </Box>
          )}

          {/* モード別の画像 or 図表を描画 */}
          <Box mt={2}>
            {/* light or t1contour → Scatterで輪郭表示 */}
            {drawMode === "light" && <Scatter data={contourPlotData} options={contourPlotOptions} />}
            {drawMode === "t1contour" && <Scatter data={contourPlotData} options={contourPlotOptions} />}

            {/* replot画像 */}
            {drawMode === "replot" && images[cellIds[currentIndex]]?.replot && (
              <img
                src={images[cellIds[currentIndex]]?.replot}
                alt={`Cell ${cellIds[currentIndex]} Replot`}
                style={{ width: "100%" }}
              />
            )}

            {/* distribution画像 (非正規化) */}
            {drawMode === "distribution" && images[cellIds[currentIndex]]?.distribution && (
              <img
                src={images[cellIds[currentIndex]]?.distribution}
                alt={`Cell ${cellIds[currentIndex]} Distribution`}
                style={{ width: "100%" }}
              />
            )}

            {/* distribution (正規化) → distribution_normalized */}
            {drawMode === "distribution_normalized" && images[cellIds[currentIndex]]?.distribution_normalized && (
              <img
                src={images[cellIds[currentIndex]]?.distribution_normalized}
                alt={`Cell ${cellIds[currentIndex]} Distribution (Normalized)`}
                style={{ width: "100%" }}
              />
            )}

            {/* peak-path画像 */}
            {drawMode === "path" && isLoading ? (
              <Box display="flex" justifyContent="center" alignItems="center" style={{ height: 400 }}>
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

            {/* prediction画像 */}
            {drawMode === "prediction" && images[cellIds[currentIndex]]?.prediction && (
              <img
                src={images[cellIds[currentIndex]]?.prediction}
                alt={`Cell ${cellIds[currentIndex]} Prediction`}
                style={{ width: "100%" }}
              />
            )}

            {/* 3D Fluo */}
            {drawMode === "cloud_points" && images[cellIds[currentIndex]]?.cloud_points && (
              <img
                src={images[cellIds[currentIndex]]?.cloud_points}
                alt={`Cell ${cellIds[currentIndex]} 3D`}
                style={{ width: "100%" }}
              />
            )}

            {/* 3D PH */}
            {drawMode === "cloud_points_ph" && images[cellIds[currentIndex]]?.cloud_points_ph && (
              <img
                src={images[cellIds[currentIndex]]?.cloud_points_ph}
                alt={`Cell ${cellIds[currentIndex]} 3D`}
                style={{ width: "100%" }}
              />
            )}
          </Box>
        </Box>

        {/* 右カラム: MorphoEngine (None, Median, Mean など) */}
        <Box sx={{ width: 350, height: 420, marginLeft: 2 }}>
          <FormControl fullWidth>
            <InputLabel id="engine-select-label">MorphoEngine</InputLabel>
            <Select
              labelId="engine-select-label"
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
                      <img src={logoPath} alt="" style={{ width: 24, height: 24, marginRight: 8 }} />
                    )}
                    {engine}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Noneのとき → ロゴ表示 */}
          {engineMode === "None" && (
            <Box
              sx={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                height: "100%",
              }}
            >
              <Typography variant="h5">MorphoEngine is off.</Typography>
              <img src="/logo_tp.png" alt="Morpho Engine is off" style={{ maxWidth: "15%", maxHeight: "15%" }} />
            </Box>
          )}

          {/* MorphoEngine 2.0 */}
          {engineMode === "MorphoEngine 2.0" && (
            <Box mt={2}>
              <CellMorphologyTable
                cellId={cellIds[currentIndex]}
                db_name={db_name}
                polyfitDegree={fitDegree}
              />
            </Box>
          )}

          {/* MedianEngine */}
          {engineMode === "MedianEngine" && (
            <Box mt={6}>
              <MedianEngine dbName={db_name} label={selectedLabel} cellId={cellIds[currentIndex]} />
            </Box>
          )}

          {/* MeanEngine */}
          {engineMode === "MeanEngine" && (
            <Box mt={6}>
              <MeanEngine dbName={db_name} label={selectedLabel} cellId={cellIds[currentIndex]} />
            </Box>
          )}

          {/* VarEngine */}
          {engineMode === "VarEngine" && (
            <Box mt={6}>
              <VarEngine dbName={db_name} label={selectedLabel} cellId={cellIds[currentIndex]} />
            </Box>
          )}

          {/* HeatmapEngine */}
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
        </Box>
      </Stack>
    </>
  );
>>>>>>> main
};

export default CellImageGrid;
