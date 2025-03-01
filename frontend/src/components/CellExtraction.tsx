import React, { useState } from "react";
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
  CardActions,
  Divider,
} from "@mui/material";
import { styled } from "@mui/system";
import axios from "axios";
import { settings } from "../settings";
import ArrowBackIosIcon from "@mui/icons-material/ArrowBackIos";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";

const url_prefix = settings.url_prefix;

/**
 * テキストフィールドのスタイルをカスタマイズ
 * - スピンボタン非表示など
 */
const CustomTextField = styled(TextField)({
  "& .MuiOutlinedInput-root": {
    "& fieldset": {
      borderColor: "black",
    },
    "&:hover fieldset": {
      borderColor: "black",
    },
    "&.Mui-focused fieldset": {
      borderColor: "black",
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
});

interface CellExtractionResponse {
  num_tiff: number;
  ulid: string;
}

const Extraction: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const fileName = searchParams.get("file_name") || "";

  const [mode, setMode] = useState("dual_layer");
  const [param1, setParam1] = useState("100");
  const [imageSize, setImageSize] = useState("200");
  const [isLoading, setIsLoading] = useState(false);
  const [numImages, setNumImages] = useState(0);
  const [currentImage, setCurrentImage] = useState(0);
  const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);
  const [sessionUlid, setSessionUlid] = useState<string | null>(null);

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

    try {
      const extractRes = await axios.get<CellExtractionResponse>(
        `${url_prefix}/cell_extraction/${fileName}/${actualMode}`,
        {
          params: {
            param1: numericParam1,
            image_size: numericImageSize,
            reverse_layers: reverseLayers,
          },
          headers,
        }
      );
      const ulid = extractRes.data.ulid;
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

  const handleGoToDatabases = async () => {
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
              </Select>
            </FormControl>

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

            <Box mt={3}>
              <Button
                variant="contained"
                fullWidth
                onClick={handleExtractCells}
                disabled={isLoading}
                sx={{
                  backgroundColor: "black",
                  color: "white",
                  height: "56px",
                  textTransform: "none",
                  "&:hover": {
                    backgroundColor: "grey",
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
                    backgroundColor: "black",
                    color: "white",
                    height: "56px",
                    textTransform: "none",
                    "&:hover": {
                      backgroundColor: "grey",
                    },
                  }}
                >
                  Go to Databases
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* 右カラム: 画像部分 */}
        {currentImageUrl && (
          <Grid item xs={12} md={8}>
            <Card sx={{ display: "flex", flexDirection: "column" }}>
              <CardContent
                sx={{ display: "flex", flexDirection: "column", alignItems: "center" }}
              >
                <Box
                  component="img"
                  src={currentImageUrl}
                  alt={`Extracted cell ${currentImage}`}
                  sx={{ width: "100%", maxWidth: 400, height: 400, objectFit: "contain" }}
                />
              </CardContent>
              <CardActions
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  px: 2,
                  pb: 2,
                }}
              >
                <Button
                  variant="contained"
                  onClick={handlePreviousImage}
                  disabled={currentImage === 0}
                  startIcon={<ArrowBackIosIcon />}
                  sx={{
                    backgroundColor: "black",
                    color: "white",
                    textTransform: "none",
                    "&:hover": {
                      backgroundColor: "grey",
                    },
                  }}
                >
                  Previous
                </Button>
                <Typography variant="body1">
                  {currentImage + 1} / {numImages}
                </Typography>
                <Button
                  variant="contained"
                  onClick={handleNextImage}
                  disabled={currentImage === numImages - 1}
                  endIcon={<ArrowForwardIosIcon />}
                  sx={{
                    backgroundColor: "black",
                    color: "white",
                    textTransform: "none",
                    "&:hover": {
                      backgroundColor: "grey",
                    },
                  }}
                >
                  Next
                </Button>
              </CardActions>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Extraction;
