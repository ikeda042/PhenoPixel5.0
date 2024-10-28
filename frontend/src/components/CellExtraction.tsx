import React, { useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import {
    Box, Grid, Typography, TextField, Button, MenuItem, Select, FormControl, InputLabel, Backdrop, CircularProgress, Breadcrumbs, Link
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

interface CellExtractionResponse {
    num_tiff: number;
    ulid: string;
}

const Extraction: React.FC = () => {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const fileName = searchParams.get("file_name") || "";
    const [mode, setMode] = useState("dual_layer");
    const [param1, setParam1] = useState(100);
    const [imageSize, setImageSize] = useState(200);
    const [isLoading, setIsLoading] = useState(false);
    const [numImages, setNumImages] = useState(0);
    const [currentImage, setCurrentImage] = useState(0);
    const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);
    const [sessionUlid, setSessionUlid] = useState<string | null>(null);

    const handleExtractCells = async () => {
        setIsLoading(true);
        const reverseLayers = mode === "dual_layer_reversed";
        const actualMode = mode === "dual_layer_reversed" ? "dual_layer" : mode;

        try {
            const extractResponse = await axios.get<CellExtractionResponse>(`${url_prefix}/cell_extraction/${fileName}/${actualMode}`, {
                params: {
                    param1,
                    image_size: imageSize,
                    reverse_layers: reverseLayers,
                },
            });
            const { ulid } = extractResponse.data;
            setSessionUlid(ulid);

            const countResponse = await axios.get(`${url_prefix}/cell_extraction/ph_contours/${ulid}/count`);
            const numImages = countResponse.data.count;
            setNumImages(numImages);
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
            const response = await axios.get(`${url_prefix}/cell_extraction/ph_contours/${ulid}/${frameNum}`, {
                responseType: 'blob',
            });
            const imageUrl = URL.createObjectURL(response.data);
            setCurrentImageUrl(imageUrl);
        } catch (error) {
            console.error("Failed to fetch image", error);
        }
    };

    const handlePreviousImage = () => {
        const newImage = currentImage - 1;
        if (newImage >= 0 && sessionUlid) {
            setCurrentImage(newImage);
            fetchImage(newImage, sessionUlid);
        }
    };

    const handleNextImage = () => {
        const newImage = currentImage + 1;
        if (newImage < numImages && sessionUlid) {
            setCurrentImage(newImage);
            fetchImage(newImage, sessionUlid);
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
                                <MenuItem value="dual_layer_reversed">Dual Layer (Reversed)</MenuItem> {/* Use reverse_layers for this */}
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
                    </Box>
                </Grid>
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
                                    }
                                    }
                                >
                                    Next
                                </Button>
                            </Box>
                        </Box>
                    </Grid>
                )}
            </Grid>
        </Box>
    );
};

export default Extraction;