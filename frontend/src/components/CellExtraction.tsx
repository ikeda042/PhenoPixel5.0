import React, { useMemo, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import {
  Box,
  Grid,
  Typography,
  TextField,
  Button,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Backdrop,
  CircularProgress,
  Breadcrumbs,
  Link,
  Paper,
  Card,
  CardContent,
  Divider,
  IconButton,
  InputAdornment,
  Slider,
  Stack,
  Chip,
} from "@mui/material";
import { alpha } from "@mui/material/styles";
import { styled } from "@mui/system";
import axios from "axios";
import { settings } from "../settings";
import ArrowBackIosIcon from "@mui/icons-material/ArrowBackIos";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import AddCircleOutlineIcon from "@mui/icons-material/AddCircleOutline";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";

const url_prefix = settings.url_prefix;

/**
 * テキストフィールドのスタイルをカスタマイズ
 * - スピンボタン非表示など
 */
const CustomTextField = styled(TextField)(({ theme }) => ({
  "& .MuiOutlinedInput-root": {
    "& fieldset": {
      borderColor: theme.palette.text.primary,
    },
    "&:hover fieldset": {
      borderColor: theme.palette.text.primary,
    },
    "&.Mui-focused fieldset": {
      borderColor: theme.palette.text.primary,
    },
  },
  "& input[type=number]": {
    "-moz-appearance": "textfield",
  },
  "& input[type=number]::-webkit-outer-spin-button": {
    "-webkit-appearance": "none",
    margin: 0,
  },
  "& input[type=number]::-webkit-inner-spin-button": {
    "-webkit-appearance": "none",
    margin: 0,
  },
}));

interface CreatedDatabase {
  frame_start: number;
  frame_end: number;
  db_name: string;
}

interface CellExtractionResponse {
  num_tiff: number;
  ulid: string;
  db_name: string;
  created_databases?: CreatedDatabase[];
}

interface SplitFrameRow {
  frameStart: string;
  frameEnd: string;
  dbName: string;
}

const Extraction: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const fileName = searchParams.get("file_name") || "";
  const nd2Stem = useMemo(() => {
    if (!fileName) {
      return "";
    }
    const lastDot = fileName.lastIndexOf(".");
    if (lastDot > 0) {
      return fileName.slice(0, lastDot);
    }
    return fileName;
  }, [fileName]);
  const dbNamePrefix = nd2Stem ? `${nd2Stem}-` : "";

  const [mode, setMode] = useState("dual_layer");
  const [param1, setParam1] = useState("100");
  const [imageSize, setImageSize] = useState("200");
  const [isLoading, setIsLoading] = useState(false);
  const [numImages, setNumImages] = useState(0);
  const [currentImage, setCurrentImage] = useState(0);
  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);
  const [sessionUlid, setSessionUlid] = useState<string | null>(null);
  const [createdDbName, setCreatedDbName] = useState<string | null>(null);
  const [createdDatabases, setCreatedDatabases] = useState<CreatedDatabase[]>([]);
  const [splitFrames, setSplitFrames] = useState<SplitFrameRow[]>([]);
  const [splitFramesError, setSplitFramesError] = useState<string | null>(null);
  const sliderMax = Math.max(numImages - 1, 0);

  const displayDbName = useMemo(() => {
    if (createdDatabases.length > 0) {
      const [first, ...rest] = createdDatabases;
      if (rest.length > 0) {
        return `${first.db_name} (+${rest.length} more)`;
      }
      return first.db_name;
    }
    if (createdDbName) {
      return createdDbName;
    }
    return "Not saved yet";
  }, [createdDbName, createdDatabases]);

  /**
   * param1, imageSize の入力中の先頭0を除去するためのハンドラ
   */
  const handleBlurNumericInput = (
    e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>,
    setState: React.Dispatch<React.SetStateAction<string>>
  ) => {
    const numValue = parseFloat(e.target.value);
    if (isNaN(numValue)) {
      setState("0");
    } else {
      setState(String(numValue));
    }
  };

  const handleAddSplitFrame = () => {
    setSplitFrames((prev) => [...prev, { frameStart: "", frameEnd: "", dbName: "" }]);
  };

  const handleSplitFrameChange = (
    index: number,
    field: keyof SplitFrameRow,
    value: string
  ) => {
    setSplitFrames((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], [field]: value };
      return next;
    });
  };

  const handleRemoveSplitFrame = (index: number) => {
    setSplitFrames((prev) => prev.filter((_, i) => i !== index));
  };

  const handleExtractCells = async () => {
    setIsLoading(true);
    const reverseLayers = mode === "dual_layer_reversed";
    const actualMode = reverseLayers ? "dual_layer" : mode;

    // param1, imageSizeを数値化
    const numericParam1 = parseFloat(param1);
    const numericImageSize = parseFloat(imageSize);

    // localStorage に保存されたアクセストークンを取得
    const token = localStorage.getItem("access_token");
    const headers = token ? { Authorization: `Bearer ${token}` } : {};

    const populatedSplits = splitFrames.filter(
      (split) =>
        split.frameStart !== "" ||
        split.frameEnd !== "" ||
        split.dbName.trim() !== ""
    );
    let splitFramesPayload: { frame_start: number; frame_end: number; db_name: string }[] | null =
      null;

    if (populatedSplits.length > 0) {
      const hasIncomplete = populatedSplits.some(
        (split) => split.frameStart === "" || split.frameEnd === "" || split.dbName.trim() === ""
      );
      if (hasIncomplete) {
        setSplitFramesError("Split frames: fill every field or remove the row.");
        setIsLoading(false);
        return;
      }
      const formatted = populatedSplits.map((split) => {
        const frame_start = parseInt(split.frameStart, 10);
        const frame_end = parseInt(split.frameEnd, 10);
        const suffix = split.dbName.trim();
        const prefixedName = dbNamePrefix ? `${dbNamePrefix}${suffix}` : suffix;
        return {
          frame_start,
          frame_end,
          db_name: prefixedName,
        };
      });
      const hasInvalidNumber = formatted.some(
        (split) => Number.isNaN(split.frame_start) || Number.isNaN(split.frame_end)
      );
      if (hasInvalidNumber) {
        setSplitFramesError("Frame start/end must be valid numbers.");
        setIsLoading(false);
        return;
      }
      const hasNegative = formatted.some(
        (split) => split.frame_start < 0 || split.frame_end < 0
      );
      if (hasNegative) {
        setSplitFramesError("Frame start/end must be 0 or greater.");
        setIsLoading(false);
        return;
      }
      splitFramesPayload = formatted;
      setSplitFramesError(null);
    } else {
      setSplitFramesError(null);
    }

    try {
      setCreatedDbName(null);
      setCreatedDatabases([]);
      const queryParams: Record<string, string | number | boolean> = {
        param1: numericParam1,
        image_size: numericImageSize,
        reverse_layers: reverseLayers,
      };
      if (splitFramesPayload) {
        queryParams.split_frames = JSON.stringify(splitFramesPayload);
      }
      const extractRes = await axios.get<CellExtractionResponse>(
        `${url_prefix}/cell_extraction/${fileName}/${actualMode}`,
        {
          params: queryParams,
          headers,
        }
      );
      const ulid = extractRes.data.ulid;
      setCreatedDbName(extractRes.data.db_name);
      setCreatedDatabases(extractRes.data.created_databases ?? []);
      setSessionUlid(ulid);

      const countRes = await axios.get(`${url_prefix}/cell_extraction/ph_contours/${ulid}/count`);
      const totalImages = countRes.data.count;
      setNumImages(totalImages);

      setCurrentImage(0);
      fetchImage(0, ulid);
    } catch (error) {
      console.error("Failed to extract cells", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchImage = async (frameNum: number, ulid: string) => {
    try {
      const res = await axios.get(`${url_prefix}/cell_extraction/ph_contours/${ulid}/${frameNum}`, {
        responseType: "blob",
      });
      const imageUrl = URL.createObjectURL(res.data);
      setCurrentImageUrl(imageUrl);
    } catch (error) {
      console.error("Failed to fetch image", error);
    }
  };

  const handlePreviousImage = () => {
    if (currentImage > 0 && sessionUlid) {
      const newIndex = currentImage - 1;
      setCurrentImage(newIndex);
      fetchImage(newIndex, sessionUlid);
    }
  };

  const handleNextImage = () => {
    if (currentImage < numImages - 1 && sessionUlid) {
      const newIndex = currentImage + 1;
      setCurrentImage(newIndex);
      fetchImage(newIndex, sessionUlid);
    }
  };

  const handleSliderChange = (_event: Event, value: number | number[]) => {
    if (typeof value === "number") {
      setCurrentImage(value);
    }
  };

  const handleSliderCommit = (_event: Event, value: number | number[]) => {
    if (typeof value === "number" && sessionUlid) {
      setCurrentImage(value);
      fetchImage(value, sessionUlid);
    }
  };

  const handleGoToDatabases = async () => {
    const dbsToNotify =
      createdDatabases.length > 0
        ? createdDatabases.map((db) => db.db_name)
        : createdDbName
        ? [createdDbName]
        : [];
    if (dbsToNotify.length > 0) {
      try {
        const token = localStorage.getItem("access_token");
        const headers = token ? { Authorization: `Bearer ${token}` } : {};
        await Promise.all(
          dbsToNotify.map((db) =>
            axios.post(
              `${url_prefix}/cell_extraction/databases/${db}/notify_created`,
              null,
              { headers }
            )
          )
        );
      } catch (error) {
        console.error("Failed to notify Slack", error);
      }
    }
    if (sessionUlid) {
      try {
        await axios.delete(`${url_prefix}/cell_extraction/ph_contours_delete/${sessionUlid}`);
        console.log("Files deleted successfully");
      } catch (error) {
        console.error("Failed to delete files", error);
      }
    }
    navigate(`/dbconsole?default_search_word=${fileName.slice(0, -10)}`);
  };

  /**
   * number型のinputでホイール操作が行われたときに値が変わらないようにする
   */
  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    event.currentTarget.blur();
    event.preventDefault();
  };

  return (
    <Box>
      <Backdrop open={isLoading} style={{ zIndex: 1201 }}>
        <CircularProgress color="inherit" />
      </Backdrop>
      <Box mb={2}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Link underline="hover" color="inherit" href="/nd2files">
            ND2 files
          </Link>
          <Typography color="text.primary">Cell extraction</Typography>
        </Breadcrumbs>
      </Box>

      {/* タイトルセクション */}
      <Box mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Cell Extraction
        </Typography>
        <Typography variant="body2" color="text.secondary" mt={1}>
          Extract cells from ND2 files and preview the results.
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* 左カラム: フォーム部分 */}
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" fontWeight="bold" mb={2}>
              Extraction Settings
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="body1" mb={1}>
              ND2 filename: {fileName}
            </Typography>
            <FormControl fullWidth margin="normal">
              <InputLabel>Mode</InputLabel>
              <Select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                label="Mode"
            >
              <MenuItem value="single_layer">Single Layer</MenuItem>
              <MenuItem value="dual_layer">Dual Layer</MenuItem>
              <MenuItem value="dual_layer_reversed">Dual Layer (Reversed)</MenuItem>
              <MenuItem value="triple_layer">Triple Layer</MenuItem>
              <MenuItem value="quad_layer">Quad Layer</MenuItem>
            </Select>
          </FormControl>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              画面暗め: 60-75を試す。
            </Typography>
            <Typography variant="body2" color="text.secondary">
              画面明るめ: 115-140を試す。
            </Typography>

            <CustomTextField
              label="Param1"
              type="number"
              placeholder="1-255"
              value={param1}
              onChange={(e) => setParam1(e.target.value)}
              onBlur={(e) => handleBlurNumericInput(e, setParam1)}
              onWheel={handleWheel}
              InputProps={{
                inputProps: { min: 0.1, step: 0.1 },
                autoComplete: "off",
              }}
              fullWidth
              margin="normal"
            />
            <CustomTextField
              label="Image Size"
              type="number"
              value={imageSize}
              onChange={(e) => setImageSize(e.target.value)}
              onBlur={(e) => handleBlurNumericInput(e, setImageSize)}
              onWheel={handleWheel}
              fullWidth
              margin="normal"
            />

            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle1" fontWeight="bold">
              Split frames (optional)
            </Typography>
            <Typography variant="body2" color="text.secondary" mb={1}>
              Create additional databases by specifying frame ranges.
            </Typography>
            {dbNamePrefix && (
              <Typography variant="body2" color="text.secondary" mb={1}>
                Saved database names will be prefixed with "{dbNamePrefix}".
              </Typography>
            )}
            {splitFrames.map((split, index) => (
              <Grid
                container
                spacing={1}
                alignItems="center"
                key={`split-${index}`}
                sx={{ mb: 1 }}
              >
                <Grid item xs={12} sm={1}>
                  <CustomTextField
                    label="Start"
                    type="number"
                    value={split.frameStart}
                    onChange={(e) => handleSplitFrameChange(index, "frameStart", e.target.value)}
                    onWheel={handleWheel}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} sm={1}>
                  <CustomTextField
                    label="End"
                    type="number"
                    value={split.frameEnd}
                    onChange={(e) => handleSplitFrameChange(index, "frameEnd", e.target.value)}
                    onWheel={handleWheel}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} sm={9}>
                  <TextField
                    label="Database name"
                    value={split.dbName}
                    onChange={(e) => handleSplitFrameChange(index, "dbName", e.target.value)}
                    fullWidth
                    InputProps={{
                      startAdornment: dbNamePrefix ? (
                        <InputAdornment position="start">{dbNamePrefix}</InputAdornment>
                      ) : undefined,
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={1} sx={{ textAlign: { xs: "right", sm: "center" } }}>
                  <IconButton
                    aria-label="Remove split row"
                    color="error"
                    onClick={() => handleRemoveSplitFrame(index)}
                    size="small"
                  >
                    <DeleteOutlineIcon fontSize="small" />
                  </IconButton>
                </Grid>
              </Grid>
            ))}
            <Button
              variant="outlined"
              size="small"
              startIcon={<AddCircleOutlineIcon />}
              onClick={handleAddSplitFrame}
            >
              Add split range
            </Button>
            {splitFramesError && (
              <Typography variant="caption" color="error" display="block" mt={1}>
                {splitFramesError}
              </Typography>
            )}

            <Box mt={3}>
              <Button
                variant="contained"
                fullWidth
                onClick={handleExtractCells}
                disabled={isLoading}
                sx={{
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  height: '56px',
                  textTransform: 'none',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                }}
              >
                {numImages > 0 ? "Re-extract Cells" : "Extract Cells"}
              </Button>
            </Box>
            {numImages > 0 && (
              <Box mt={2}>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={handleGoToDatabases}
                  sx={{
                    backgroundColor: 'primary.main',
                    color: 'primary.contrastText',
                    height: '56px',
                    textTransform: 'none',
                    '&:hover': {
                      backgroundColor: 'primary.dark',
                    },
                  }}
                >
                  Go to Databases
                </Button>
              </Box>
            )}
            {createdDatabases.length > 0 && (
              <Box mt={2}>
                <Typography variant="subtitle2" fontWeight="bold">
                  Created databases
                </Typography>
                {createdDatabases.map((db, index) => (
                  <Typography variant="body2" key={`${db.db_name}-${index}`}>
                    {db.db_name} (frames {db.frame_start} - {db.frame_end})
                  </Typography>
                ))}
              </Box>
            )}
          </Paper>
        </Grid>

        {/* 右カラム: 画像部分 */}
        {currentImageUrl && (
          <Grid item xs={12} md={8}>
            <Card
              sx={(theme) => ({
                display: "flex",
                flexDirection: "column",
                borderRadius: 3,
                overflow: "hidden",
                boxShadow: 10,
                background:
                  theme.palette.mode === "dark"
                    ? "linear-gradient(135deg, #0b1224 0%, #0f172a 55%, #0b1020 100%)"
                    : "linear-gradient(135deg, #f8fafc 0%, #eef2ff 55%, #e0f2fe 100%)",
              })}
            >
              <CardContent
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 2.5,
                  p: { xs: 2.5, sm: 3.5 },
                }}
              >
                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: { xs: "1fr", lg: "1.2fr 0.8fr" },
                    gap: { xs: 2, sm: 3 },
                    alignItems: "stretch",
                  }}
                >
                  <Box
                    sx={(theme) => ({
                      position: "relative",
                      borderRadius: 3,
                      overflow: "hidden",
                      minHeight: { xs: 280, sm: 380 },
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      background:
                        theme.palette.mode === "dark"
                          ? "radial-gradient(circle at 20% 20%, rgba(255,255,255,0.06), transparent 35%), linear-gradient(145deg, #0f172a 0%, #0b1224 50%, #090f1d 100%)"
                          : "radial-gradient(circle at 20% 20%, rgba(79,70,229,0.14), transparent 32%), linear-gradient(145deg, #eef2ff 0%, #e0f2fe 50%, #f8fafc 100%)",
                      border: `1px solid ${
                        theme.palette.mode === "dark"
                          ? alpha(theme.palette.primary.main, 0.25)
                          : alpha(theme.palette.text.primary, 0.08)
                      }`,
                      boxShadow: theme.shadows[6],
                    })}
                  >
                    <Chip
                      label="PH + Contour"
                      size="small"
                      color="primary"
                      sx={{
                        position: "absolute",
                        top: 16,
                        left: 16,
                        fontWeight: 700,
                        letterSpacing: 0.5,
                      }}
                    />
                    <Box
                      component="img"
                      src={currentImageUrl}
                      alt={`Extracted cell ${currentImage + 1}`}
                      sx={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        mixBlendMode: "screen",
                      }}
                    />
                    <Box
                      sx={(theme) => ({
                        position: "absolute",
                        bottom: 16,
                        right: 16,
                        px: 1.5,
                        py: 0.75,
                        borderRadius: 2,
                        backdropFilter: "blur(8px)",
                        backgroundColor:
                          theme.palette.mode === "dark"
                            ? "rgba(15, 23, 42, 0.7)"
                            : "rgba(255, 255, 255, 0.85)",
                        border: `1px solid ${
                          theme.palette.mode === "dark"
                            ? "rgba(255,255,255,0.14)"
                            : "rgba(0,0,0,0.08)"
                        }`,
                      })}
                    >
                      <Typography variant="body2" fontWeight="bold">
                        Frame {currentImage + 1} / {numImages}
                      </Typography>
                    </Box>
                  </Box>

                  <Stack
                    spacing={2}
                    sx={(theme) => ({
                      borderRadius: 3,
                      p: { xs: 2, sm: 2.5 },
                      backgroundColor:
                        theme.palette.mode === "dark"
                          ? "rgba(255, 255, 255, 0.02)"
                          : "rgba(0, 0, 0, 0.02)",
                      border: `1px solid ${
                        theme.palette.mode === "dark"
                          ? "rgba(255,255,255,0.08)"
                          : "rgba(0,0,0,0.05)"
                      }`,
                      boxShadow: theme.shadows[2],
                    })}
                  >
                    <Typography
                      variant="overline"
                      color="text.secondary"
                      letterSpacing={1}
                      fontWeight={700}
                    >
                      Extraction summary
                    </Typography>
                    <Box>
                      <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                        {fileName || "No ND2 selected"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Mode: {mode.replace(/_/g, " ")}
                      </Typography>
                    </Box>
                    <Divider />
                    <Grid container spacing={1.5}>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Param1
                        </Typography>
                        <Typography variant="body1" fontWeight="bold">
                          {param1}
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="caption" color="text.secondary">
                          Image size
                        </Typography>
                        <Typography variant="body1" fontWeight="bold">
                          {imageSize}px
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="caption" color="text.secondary">
                          Database
                        </Typography>
                        <Typography variant="body1" fontWeight="bold">
                          {displayDbName}
                        </Typography>
                      </Grid>
                    </Grid>
                    {createdDatabases.length > 1 && (
                      <Typography variant="caption" color="text.secondary">
                        +{createdDatabases.length - 1} more split database(s) prepared
                      </Typography>
                    )}
                  </Stack>
                </Box>

                <Divider />

                <Stack
                  spacing={2}
                  direction={{ xs: "column", sm: "row" }}
                  alignItems="center"
                >
                  <Button
                    variant="contained"
                    onClick={handlePreviousImage}
                    disabled={currentImage === 0}
                    startIcon={<ArrowBackIosIcon />}
                    sx={{
                      minWidth: 140,
                      backgroundColor: 'primary.main',
                      color: 'primary.contrastText',
                      textTransform: 'none',
                      '&:hover': {
                        backgroundColor: 'primary.dark',
                      },
                    }}
                  >
                    Previous
                  </Button>

                  <Box sx={{ flexGrow: 1, width: "100%" }}>
                    <Typography variant="caption" color="text.secondary">
                      Frame navigator
                    </Typography>
                    <Slider
                      size="small"
                      value={Math.min(currentImage, sliderMax)}
                      min={0}
                      max={sliderMax}
                      step={1}
                      onChange={handleSliderChange}
                      onChangeCommitted={handleSliderCommit}
                      valueLabelDisplay="auto"
                      valueLabelFormat={(value) => `${value + 1}`}
                      disabled={numImages <= 1}
                    />
                  </Box>

                  <Button
                    variant="contained"
                    onClick={handleNextImage}
                    disabled={currentImage === numImages - 1}
                    endIcon={<ArrowForwardIosIcon />}
                    sx={{
                      minWidth: 140,
                      backgroundColor: 'primary.main',
                      color: 'primary.contrastText',
                      textTransform: 'none',
                      '&:hover': {
                        backgroundColor: 'primary.dark',
                      },
                    }}
                  >
                    Next
                  </Button>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Extraction;
