import React, { useState } from "react";
import { Box, Button, Container, Typography, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent, CircularProgress, Modal } from "@mui/material";
import { settings } from "../settings";
import { Breadcrumbs, Link } from "@mui/material";

const url_prefix = settings.url_prefix;
const GraphEngine: React.FC = () => {
    const [mode, setMode] = useState("heatmap_abs");
    const [file, setFile] = useState<File | null>(null);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleModeChange = (event: SelectChangeEvent<string>) => {
        setMode(event.target.value as string);
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            setFile(event.target.files[0]);
        }
    };

    const handleGenerateGraph = async () => {
        if (!file) {
            alert("Please select a CSV file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        setIsLoading(true);

        try {
            const response = await fetch(`${url_prefix}/graph_engine/${mode}`, {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                setImageSrc(imageUrl);
            } else {
                alert("Failed to generate graph.");
            }
        } catch (error) {
            console.error("Error generating graph:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const modalStyle = {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    };

    return (
        <Container>
            <Box mb={3}>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Typography color="text.primary">Graph engine</Typography>
                </Breadcrumbs>
            </Box>
            <FormControl fullWidth>
                <InputLabel id="select-label">Graph Mode</InputLabel>
                <Select
                    labelId="select-label"
                    value={mode}
                    onChange={handleModeChange}
                    style={{ color: isLoading ? "black" : undefined }}
                >
                    <MenuItem value="heatmap_abs">Heatmap abs.</MenuItem>
                    <MenuItem value="heatmap_rel">Heatmap rel.</MenuItem>
                </Select>
            </FormControl>
            <Box my={2}>
                <input type="file" accept=".csv" onChange={handleFileChange} />
            </Box>
            <Button
                variant="contained"
                color="success"
                onClick={handleGenerateGraph}
                disabled={isLoading}
                style={{ backgroundColor: isLoading ? "black" : undefined }}
            >
                {isLoading ? <CircularProgress size={24} color="inherit" /> : "Generate Graph"}
            </Button>

            <Modal
                open={isLoading}
                aria-labelledby="loading-modal"
                aria-describedby="loading-graph-generation"
                style={modalStyle}
            >
                <Box>
                    <CircularProgress color="primary" />
                </Box>
            </Modal>

            {imageSrc && (
                <Box mt={4}>
                    <Typography variant="h6">Generated Graph:</Typography>
                    <img
                        src={imageSrc}
                        alt="Generated Graph"
                        style={{
                            maxWidth: "100%",
                            maxHeight: "80vh",
                            display: "block",
                            margin: "0 auto"
                        }}
                    />
                </Box>
            )}
        </Container>
    );
};

export default GraphEngine;