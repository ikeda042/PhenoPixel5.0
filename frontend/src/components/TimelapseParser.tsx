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
    Link
} from "@mui/material";
import { styled } from '@mui/system';
import axios from "axios";
import { settings } from "../settings";
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';

const url_prefix = settings.url_prefix;

const CustomTextField = styled(TextField)({
    '& .MuiOutlinedInput-root': {
        '& fieldset': {
            borderColor: 'black',
        },
        '&:hover fieldset': {
            borderColor: 'black',
        },
        '&.Mui-focused fieldset': {
            borderColor: 'black',
        },
    },
    '& input[type=number]': {
        '-moz-appearance': 'textfield',
    },
    '& input[type=number]::-webkit-outer-spin-button': {
        '-webkit-appearance': 'none',
        margin: 0,
    },
    '& input[type=number]::-webkit-inner-spin-button': {
        '-webkit-appearance': 'none',
        margin: 0,
    },
});

const TimelapseParser: React.FC = () => {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const fileName = searchParams.get("file_name") || "";

    // 追加パラメータや状態管理
    const [mode, setMode] = useState("dual_layer");
    const [param1, setParam1] = useState(100);
    const [imageSize, setImageSize] = useState(200);
    const [isLoading, setIsLoading] = useState(false);

    // 既存のイメージ表示用のステート
    const [numImages, setNumImages] = useState(0);
    const [currentImage, setCurrentImage] = useState(0);
    const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);

    // --- ここから追加 ---
    // パース結果から得られる「Field」一覧と、選択中のField
    const [fields, setFields] = useState<string[]>([]);
    const [selectedField, setSelectedField] = useState<string>("");
    // 選択したFieldのGIF URL
    const [gifUrl, setGifUrl] = useState<string>("");

    // パース実行ボタン
    const handleExtractCells = async () => {
        setIsLoading(true);
        const reverseLayers = mode === "dual_layer_reversed";
        try {
            // 1) タイムラプスND2ファイルの解析を実行
            //    (サーバー側で解析が終わったら、fieldsの一覧を返すようにしている想定)
            const parseResponse = await axios.get(`${url_prefix}/tlengine/nd2_files/${fileName}`, {
                params: {
                    param1,
                    image_size: imageSize,
                    reverse_layers: reverseLayers,
                },
            });
            // parseResponse.data の中に { fields: string[] } が含まれている想定
            if (parseResponse.data.fields) {
                setFields(parseResponse.data.fields);
            } else {
                // 仮にサーバーから data.fields が無い場合は、空配列に
                setFields([]);
            }

            // 2) 既存の細胞画像抽出数を取得
            const countResponse = await axios.get(`${url_prefix}/cell_extraction/ph_contours/count`);
            const numImages = countResponse.data.count;
            setNumImages(numImages);
            setCurrentImage(0);
            // 最初の画像を取得
            fetchImage(0);
        } catch (error) {
            console.error("Failed to extract cells", error);
        } finally {
            setIsLoading(false);
        }
    };

    // 画像の取得
    const fetchImage = async (frameNum: number) => {
        try {
            const response = await axios.get(`${url_prefix}/cell_extraction/ph_contours/${frameNum}`, {
                responseType: 'blob',
            });
            const imageUrl = URL.createObjectURL(response.data);
            setCurrentImageUrl(imageUrl);
        } catch (error) {
            console.error("Failed to fetch image", error);
        }
    };

    // GIFを取得
    // Fieldを選んだタイミングなどで呼び出して、GIFを表示用にblob化
    const fetchGif = async (fieldValue: string) => {
        if (!fileName || !fieldValue) return;
        try {
            setIsLoading(true);
            const response = await axios.get(
                `${url_prefix}/tlengine/nd2_files/${fileName}/gif/${fieldValue}`,
                { responseType: "blob" }
            );
            const blobUrl = URL.createObjectURL(response.data);
            setGifUrl(blobUrl);
        } catch (error) {
            console.error("Failed to fetch GIF", error);
        } finally {
            setIsLoading(false);
        }
    };

    // 視野(Field)が選択されたとき
    const handleFieldChange = (event: React.ChangeEvent<{ value: unknown }>) => {
        const newField = event.target.value as string;
        setSelectedField(newField);
        // Field 選択時にGIFを取得
        fetchGif(newField);
    };

    // 既存の「前へ」「次へ」処理
    const handlePreviousImage = () => {
        const newImage = currentImage - 1;
        if (newImage >= 0) {
            setCurrentImage(newImage);
            fetchImage(newImage);
        }
    };

    const handleNextImage = () => {
        const newImage = currentImage + 1;
        if (newImage < numImages) {
            setCurrentImage(newImage);
            fetchImage(newImage);
        }
    };

    const handleGoToDatabases = () => {
        navigate(`/dbconsole?default_search_word=${fileName.slice(0, -10)}`);
    };

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
                    {/* ここも tlengine にあわせて修正 */}
                    <Link underline="hover" color="inherit" href="/tlengine">
                        Timelapse ND2 files
                    </Link>
                    <Typography color="text.primary">Cell extraction</Typography>
                </Breadcrumbs>
            </Box>
            <Grid container spacing={2} >
                <Grid item xs={12} md={4} style={{ display: 'flex', justifyContent: 'center' }}>
                    <Box width="100%" >
                        <Typography variant="body1">
                            nd2 filename :  {fileName}
                        </Typography>
                        <FormControl fullWidth margin="normal">
                            <InputLabel>Mode</InputLabel>
                            <Select
                                value={mode}
                                onChange={(e) => setMode(e.target.value)}
                            >
                                <MenuItem value="single_layer">Single Layer</MenuItem>
                                <MenuItem value="dual_layer">Dual Layer</MenuItem>
                                <MenuItem value="dual_layer_reversed">Dual Layer (Reversed)</MenuItem>
                                <MenuItem value="triple_layer">Triple Layer</MenuItem>
                            </Select>
                        </FormControl>
                        <TextField
                            label="Param1"
                            type="number"
                            placeholder="1-255"
                            value={param1}
                            onChange={(e) => setParam1(Number(e.target.value))}
                            InputProps={{
                                inputProps: { min: 0.1, step: 0.1 },
                                onWheel: handleWheel,
                                autoComplete: "off"
                            }}
                        />
                        <CustomTextField
                            label="Image Size"
                            type="number"
                            fullWidth
                            margin="normal"
                            value={imageSize}
                            onChange={(e) => setImageSize(Number(e.target.value))}
                        />
                        <Button
                            variant="contained"
                            color="primary"
                            fullWidth
                            onClick={handleExtractCells}
                            disabled={isLoading}
                            sx={{
                                backgroundColor: 'black',
                                color: 'white',
                                height: '56px',
                                textTransform: 'none',
                                '&:hover': {
                                    backgroundColor: 'grey'
                                }
                            }}
                        >
                            {numImages > 0 && ("Re-extract Cells")}
                            {numImages === 0 && ("Extract Cells")}
                        </Button>
                        {numImages > 0 && (
                            <Button
                                variant="contained"
                                color="secondary"
                                fullWidth
                                onClick={handleGoToDatabases}
                                sx={{
                                    marginTop: 2,
                                    backgroundColor: 'black',
                                    textTransform: 'none',
                                    color: 'white',
                                    height: '56px',
                                    '&:hover': {
                                        backgroundColor: 'grey'
                                    }
                                }}
                            >
                                Go to Databases
                            </Button>
                        )}
                        {/* --- 視野(Field) の選択UIを追加 --- */}
                        {fields.length > 0 && (
                            <FormControl fullWidth margin="normal">
                                <InputLabel>Field</InputLabel>
                                <Select
                                    value={selectedField}
                                    onChange={handleFieldChange}
                                >
                                    {fields.map((field) => (
                                        <MenuItem key={field} value={field}>
                                            {field}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        )}
                    </Box>
                </Grid>
                {/* 既存のセル画像表示 */}
                {currentImageUrl && (
                    <Grid item xs={12} md={8}>
                        <Box display="flex" flexDirection="column" alignItems="center" mt={5}>
                            <Box
                                component="img"
                                src={currentImageUrl}
                                alt={`Extracted cell ${currentImage}`}
                                sx={{ width: '400px', height: '400px', objectFit: 'contain' }}
                            />
                            <Box display="flex" justifyContent="space-between" width="400px" mt={2}>
                                <Button
                                    variant="contained"
                                    onClick={handlePreviousImage}
                                    disabled={currentImage === 0}
                                    startIcon={<ArrowBackIosIcon />}
                                    sx={{
                                        backgroundColor: 'black',
                                        color: 'white',
                                        textTransform: 'none',
                                        '&:hover': {
                                            backgroundColor: 'grey'
                                        }
                                    }}
                                >
                                    Previous
                                </Button>
                                <Typography variant="body1">{currentImage + 1} / {numImages}</Typography>
                                <Button
                                    variant="contained"
                                    onClick={handleNextImage}
                                    disabled={currentImage === numImages - 1}
                                    endIcon={<ArrowForwardIosIcon />}
                                    sx={{
                                        backgroundColor: 'black',
                                        color: 'white',
                                        textTransform: 'none',
                                        '&:hover': {
                                            backgroundColor: 'grey'
                                        }
                                    }}
                                >
                                    Next
                                </Button>
                            </Box>
                        </Box>
                    </Grid>
                )}
                {/* 選択した Field の GIF 表示 */}
                {gifUrl && (
                    <Grid item xs={12} md={8}>
                        <Box display="flex" flexDirection="column" alignItems="center" mt={5}>
                            <Typography variant="body1" mb={2}>
                                Selected Field: {selectedField}
                            </Typography>
                            <Box
                                component="img"
                                src={gifUrl}
                                alt={`GIF for Field: ${selectedField}`}
                                sx={{ maxWidth: '400px', height: 'auto', objectFit: 'contain' }}
                            />
                        </Box>
                    </Grid>
                )}
            </Grid>
        </Box>
    );
};

export default TimelapseParser;
