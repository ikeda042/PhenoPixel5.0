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
  Grid,
} from "@mui/material";
import { ArrowBack, ArrowForward } from "@mui/icons-material";
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

// Chart.js に必要なプラグイン等を登録
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
  field: string;
  time: number;
  cell: number;
  area: number;
  perimeter: number;
  manual_label?: number;
  is_dead?: number;
}

// API 側が { "areas": number[] } で返却してくるので、
// フロントで下記のような変換用インターフェースを用意
interface ContourArea {
  frame: number;
  area: number;
}
interface GetContourAreasResponse {
  areas: number[]; // ← 本来は number[] だが、後で {frame, area} に変換する
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

  // 今表示中のセル情報（by_id から取得した詳細）
  const [currentCellData, setCurrentCellData] = useState<CellDataById | null>(null);

  // manual_label のセレクトボックス用
  const manualLabelOptions = ["N/A", "1", "2", "3", "4"];

  // GIF の再生タイミングを揃えるためのキー
  const [reloadKey, setReloadKey] = useState<number>(0);

  // 「All Cells」プレビュー用のモーダル管理
  const [openModal, setOpenModal] = useState<boolean>(false);
  const [loadingAllCells, setLoadingAllCells] = useState<boolean>(false);
  const [allCellsGifUrl, setAllCellsGifUrl] = useState<string>("");

  // 表示したいチャネル（ph, fluo1, fluo2） - 通常 GIF 用
  const channels = ["ph", "fluo1", "fluo2"] as const;

  // 輪郭面積（frame, area）に変換後の配列
  const [contourAreas, setContourAreas] = useState<ContourArea[]>([]);

  // ★ 描画モードの状態を管理
  type DrawMode = "ContourAreas" | "Replot";
  const [drawMode, setDrawMode] = useState<DrawMode>("ContourAreas");

  // ★ Replot 用に、ph, fluo1, fluo2 のどれかを選択できるようにする
  const [replotChannel, setReplotChannel] = useState<"ph" | "fluo1" | "fluo2">("ph");

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

      // 取得した際に、先頭のセル番号を自動的に選択
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
      const baseCellId = cells[0].cell_id;

      const detail = await fetchCellDataById(baseCellId);
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

  const handleChangeManualLabel = async (value: string) => {
    if (!dbName || !currentCellData) return;
    const patchLabel = value === "N/A" ? "N/A" : value;

    try {
      const baseCellId = currentCellData.cell_id;
      await axios.patch(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${baseCellId}/label?label=${patchLabel}`
      );
      fetchCurrentCellData();
    } catch (error) {
      console.error("Failed to update manual_label:", error);
    }
  };

  const handleChangeIsDead = async (checked: boolean) => {
    if (!dbName || !currentCellData) return;
    try {
      const baseCellId = currentCellData.cell_id;
      const isDeadValue = checked ? 1 : 0;
      await axios.patch(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${baseCellId}/dead/${isDeadValue}`
      );
      fetchCurrentCellData();
    } catch (error) {
      console.error("Failed to update is_dead:", error);
    }
  };

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

  const handlePrevCell = () => {
    if (cellNumbers.length === 0) return;
    const currentIndex = cellNumbers.indexOf(selectedCellNumber);
    if (currentIndex > 0) {
      setSelectedCellNumber(cellNumbers[currentIndex - 1]);
    }
  };

  // ★ ここで最後のセルに到達していたら次のフィールドへ自動的に移るようにする
  const handleNextCell = () => {
    if (cellNumbers.length === 0) return;
    const currentIndex = cellNumbers.indexOf(selectedCellNumber);

    // まだ最後のセルでなければ、次のセル番号に移動
    if (currentIndex < cellNumbers.length - 1) {
      setSelectedCellNumber(cellNumbers[currentIndex + 1]);
    } else {
      // 今のフィールドで最後のセルにいる
      const fieldIndex = fields.indexOf(selectedField);
      // 最後のフィールドでなければ、次のフィールドに移る
      if (fieldIndex < fields.length - 1) {
        setSelectedField(fields[fieldIndex + 1]);
        // ※ useEffect 内の fetchCellNumbers で次フィールドの最初のセルが自動的に選択される
      } else {
        // 最後のフィールドかつ最後のセル
        // 必要に応じて「もう最後です」などの挙動を入れてください
        console.log("すべてのフィールドとセルを見終わりました。");
      }
    }
  };

  useEffect(() => {
    setReloadKey((prev) => prev + 1);
  }, [dbName, selectedField, selectedCellNumber]);

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
        case "4":
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
  }, [
    currentCellData,
    handleChangeIsDead,
    handleChangeManualLabel,
    handleNextCell,
    handlePrevCell,
  ]);

  // 通常 GIF (3 枚表示する場合)
  const gifUrls = channels.map((ch) =>
    dbName
      ? `${url_prefix}/tlengine/databases/${dbName}/cells/gif/${selectedField}/${selectedCellNumber}?channel=${ch}`
      : ""
  );

  // replot 用 GIF (モードが Replot の時に表示)
  const replotGifUrl = dbName
    ? `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/replot?channel=${replotChannel}&degree=4`
    : "";

  const handlePreviewAllCells = async () => {
    if (!dbName || !selectedField) {
      console.error("DB名やFieldが未選択です。");
      return;
    }
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

  // ★ number[] → ContourArea[] に変換してから setContourAreas する
  const fetchContourAreas = async () => {
    if (!dbName || !selectedField || !selectedCellNumber) {
      setContourAreas([]);
      return;
    }
    try {
      const response = await axios.get<GetContourAreasResponse>(
        `${url_prefix}/tlengine/databases/${dbName}/cells/${selectedField}/${selectedCellNumber}/contour_areas`
      );
      // 受け取った数値配列を { frame, area } の形に変換
      const converted = response.data.areas.map((value, index) => ({
        frame: index,
        area: value,
      }));
      setContourAreas(converted);
    } catch (error) {
      console.error("Failed to fetch contour areas:", error);
      setContourAreas([]);
    }
  };

  useEffect(() => {
    fetchContourAreas();
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

  // maintainAspectRatio を false にして、親要素のサイズに合わせられるようにする
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

  return (
    <>
      <Container
        sx={{
          py: 4,
          backgroundColor: "#fff",
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

              <FormControlLabel
                control={
                  <Checkbox
                    color="error"
                    checked={currentCellData.is_dead === 1}
                    onChange={(e) => handleChangeIsDead(e.target.checked)}
                  />
                }
                label="is_dead"
              />
            </>
          )}

          <Button
            variant="contained"
            startIcon={<ArrowBack />}
            sx={{
              backgroundColor: "#000",
              color: "#fff",
              "&:hover": {
                backgroundColor: "#333",
              },
            }}
            onClick={handlePrevCell}
          >
            Prev
          </Button>
          <Button
            variant="contained"
            endIcon={<ArrowForward />}
            sx={{
              backgroundColor: "#000",
              color: "#fff",
              "&:hover": {
                backgroundColor: "#333",
              },
            }}
            onClick={handleNextCell}
          >
            Next
          </Button>

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

        {/* ★ 新たに描画モード切り替え用のSelectを追加 */}
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
              <MenuItem value="Replot">Replot</MenuItem>
            </Select>
          </FormControl>

          {/* Replot のチャンネル選択用 */}
          {drawMode === "Replot" && (
            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel id="replot-channel-label">Channel</InputLabel>
              <Select
                labelId="replot-channel-label"
                value={replotChannel}
                label="Channel"
                onChange={(e) =>
                  setReplotChannel(e.target.value as "ph" | "fluo1" | "fluo2")
                }
              >
                <MenuItem value="ph">ph</MenuItem>
                <MenuItem value="fluo1">fluo1</MenuItem>
                <MenuItem value="fluo2">fluo2</MenuItem>
              </Select>
            </FormControl>
          )}
        </Box>

        {dbName ? (
          <Card
            sx={{
              borderRadius: 2,
              boxShadow: 2,
              backgroundColor: "#fff",
              mb: 4,
            }}
          >
            <CardContent>
              <Grid
                container
                spacing={2}
                justifyContent="center"
                alignItems="flex-start"
              >
                {/* 
                  常に 3つの通常 GIF を表示
                */}
                {gifUrls.map((url, idx) => (
                  <Grid
                    item
                    xs={12}
                    md={3}
                    key={`${channels[idx]}-${reloadKey}`}
                  >
                    <CardMedia
                      component="img"
                      image={url}
                      alt={`timelapse-${channels[idx]}`}
                      sx={{
                        width: "100%",
                        borderRadius: 2,
                        objectFit: "contain",
                      }}
                    />
                  </Grid>
                ))}

                {/* 描画モード別に、ContourAreas か Replot かを表示 */}
                {drawMode === "ContourAreas" && (
                  <Grid item xs={12} md={3}>
                    <Box
                      sx={{
                        // GIFと同じように幅100%で、縦横比1:1に調整
                        position: "relative",
                        width: "100%",
                        paddingBottom: "100%",
                        borderRadius: 2,
                        overflow: "hidden",
                      }}
                    >
                      {contourAreas.length > 0 ? (
                        <Box
                          sx={{
                            position: "absolute",
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                          }}
                        >
                          <Line
                            data={contourAreasChartData}
                            options={contourAreasChartOptions}
                          />
                        </Box>
                      ) : (
                        <Typography variant="body1" mt={2}>
                          輪郭面積データがありません。
                        </Typography>
                      )}
                    </Box>
                  </Grid>
                )}

                {/* Replot モードのときに 1枚だけ replot GIF を表示 */}
                {drawMode === "Replot" && (
                  <Grid item xs={12} md={4}>
                    <CardMedia
                      component="img"
                      image={replotGifUrl}
                      alt={`replot-${replotChannel}`}
                      key={`replot-${replotChannel}-${reloadKey}`}
                      sx={{
                        width: "100%",
                        borderRadius: 2,
                        objectFit: "contain",
                      }}
                    />
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        ) : (
          <Typography variant="body1" mt={2}>
            データがありません。DB名やフィールドが正しく指定されているか確認してください。
          </Typography>
        )}
      </Container>

      {/* All Cells プレビュー用ダイアログ */}
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
