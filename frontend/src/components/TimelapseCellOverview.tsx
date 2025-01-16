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

  // 表示したいチャネル（ph, fluo1, fluo2）
  const channels = ["ph", "fluo1", "fluo2"] as const;

  // 輪郭面積（frame, area）に変換後の配列
  const [contourAreas, setContourAreas] = useState<ContourArea[]>([]);

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

  const handleNextCell = () => {
    if (cellNumbers.length === 0) return;
    const currentIndex = cellNumbers.indexOf(selectedCellNumber);
    if (currentIndex >= 0 && currentIndex < cellNumbers.length - 1) {
      setSelectedCellNumber(cellNumbers[currentIndex + 1]);
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

  const gifUrls = channels.map((ch) =>
    dbName
      ? `${url_prefix}/tlengine/databases/${dbName}/cells/gif/${selectedField}/${selectedCellNumber}?channel=${ch}`
      : ""
  );

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

  // ★ ここで number[] → ContourArea[] に変換してから setContourAreas する
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

  const contourAreasChartOptions: ChartOptions<"line"> = {
    responsive: true,
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

        {dbName ? (
          <Card
            sx={{
              borderRadius: 2,
              boxShadow: 2,
              backgroundColor: "#fff",
              mb: 4,
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
              <Grid
                container
                spacing={2}
                justifyContent="center"
                alignItems="center"
              >
                {gifUrls.map((url, idx) => (
                  <Grid item xs={12} sm={4} key={`${channels[idx]}-${reloadKey}`}>
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
              </Grid>
            </CardContent>
          </Card>
        ) : (
          <Typography variant="body1" mt={2}>
            データがありません。DB名やフィールドが正しく指定されているか確認してください。
          </Typography>
        )}

        <Card
          sx={{
            borderRadius: 2,
            boxShadow: 2,
            backgroundColor: "#fff",
          }}
        >
          <CardHeader
            title="Contour Areas"
            sx={{
              pb: 1,
              "& .MuiCardHeader-title": {
                fontWeight: "bold",
              },
            }}
          />
          <CardContent>
            {contourAreas.length > 0 ? (
              <Line
                data={contourAreasChartData}
                options={contourAreasChartOptions}
              />
            ) : (
              <Typography variant="body1" mt={2}>
                輪郭面積データがありません。
              </Typography>
            )}
          </CardContent>
        </Card>
      </Container>

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
