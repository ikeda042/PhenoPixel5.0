import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
    Box, Grid, Typography, TextField, Button, MenuItem, Select, FormControl, InputLabel, CircularProgress, IconButton
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
});

const Extraction: React.FC = () => {
    const [searchParams] = useSearchParams();
    const fileName = searchParams.get("file_name") || "";
    const [mode, setMode] = useState("dual");
    const [param1, setParam1] = useState(100);
    const [imageSize, setImageSize] = useState(200);
    const [isLoading, setIsLoading] = useState(false);
    const [numImages, setNumImages] = useState(0);
    const [currentImage, setCurrentImage] = useState(0);
    const [currentImageUrl, setCurrentImageUrl] = useState<string | null>(null);

    const handleExtractCells = async () => {
        setIsLoading(true);
        try {
            const response = await axios.get(`${url_prefix}/cell_extraction/${fileName}/${mode}`, {
                params: {
                    param1,
                    image_size: imageSize,
                },
            });

            // Fetch the number of images
            const countResponse = await axios.get(`${url_prefix}/cell_extraction/ph_contours/count`);
            const numImages = countResponse.data.count;
            setNumImages(numImages);
            setCurrentImage(0);

            // Fetch the first image
            fetchImage(0);
        } catch (error) {
            console.error("Failed to extract cells", error);
        } finally {
            setIsLoading(false);
        }
    };

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

    return (
        <Box>
            <Grid container spacing={2} alignItems="center" justifyContent="center">
                <Grid item xs={12} md={4}>
                    <FormControl fullWidth margin="normal">
                        <InputLabel>Mode</InputLabel>
                        <Select
                            value={mode}
                            onChange={(e) => setMode(e.target.value)}
                        >
                            <MenuItem value="single_layer">Single Layer</MenuItem>
                            <MenuItem value="dual_layer">Dual Layer</MenuItem>
                            <MenuItem value="triple_layer">Triple Layer</MenuItem>
                        </Select>
                    </FormControl>
                    <CustomTextField
                        label="Param1"
                        type="number"
                        fullWidth
                        margin="normal"
                        value={param1}
                        onChange={(e) => setParam1(Number(e.target.value))}
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
                            width: '100%',
                            height: '56px',
                            '&:hover': {
                                backgroundColor: 'grey'
                            }
                        }}
                    >
                        {isLoading ? <CircularProgress size={24} /> : "Extract Cells"}
                    </Button>
                </Grid>
                {currentImageUrl && (
                    <Grid item xs={12} md={8}>
                        <Box display="flex" flexDirection="column" alignItems="center">
                            <Box
                                component="img"
                                src={currentImageUrl}
                                alt={`Extracted cell ${currentImage}`}
                                sx={{ width: '400px', height: '400px', objectFit: 'contain' }}
                            />
                            <Box display="flex" justifyContent="space-between" mt={2} width="100%">
                                <IconButton onClick={handlePreviousImage} disabled={currentImage === 0}>
                                    <ArrowBackIosIcon />
                                </IconButton>
                                <Typography variant="body1">{currentImage + 1} / {numImages}</Typography>
                                <IconButton onClick={handleNextImage} disabled={currentImage === numImages - 1}>
                                    <ArrowForwardIosIcon />
                                </IconButton>
                            </Box>
                        </Box>
                    </Grid>
                )}
            </Grid>
        </Box>
    );
};

export default Extraction;
