import React, { useState } from "react";
import axios from "axios";
import {
  Box,
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import DownloadIcon from "@mui/icons-material/Download";

import { settings } from "../settings";

interface LabelOption {
  value: string;
  label: string;
}

interface PixelEngineProps {
  dbName: string;
  label: string;
  imgType: "ph" | "fluo1" | "fluo2";
  labelOptions?: LabelOption[];
  onLabelChange?: (value: string) => void;
}

const url_prefix = settings.url_prefix;

const PixelEngine: React.FC<PixelEngineProps> = ({
  dbName,
  label,
  imgType,
  labelOptions = [],
  onLabelChange,
}) => {
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);
    try {
      const response = await axios.get(
        `${url_prefix}/cells/${dbName}/${label}/pixel_engine/csv?img_type=${imgType}`,
        { responseType: "blob" }
      );
      const blob = new Blob([response.data], { type: "text/csv" });
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      const labelForName = label === "74" ? "all" : label === "1000" ? "NA" : label;
      link.href = downloadUrl;
      link.setAttribute(
        "download",
        `${dbName}_label_${labelForName}_${imgType}_pixels.csv`
      );
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error("Failed to export pixel intensities:", error);
    } finally {
      setIsExporting(false);
    }
  };

  const handleLabelSelect = (event: SelectChangeEvent<string>) => {
    const newValue = event.target.value as string;
    onLabelChange?.(newValue);
  };

  return (
    <Box display="flex" flexDirection="column" gap={2}>
      {labelOptions.length > 0 && onLabelChange && (
        <FormControl fullWidth size="small">
          <InputLabel id="pixel-engine-label-select">Label</InputLabel>
          <Select
            labelId="pixel-engine-label-select"
            value={label}
            label="Label"
            onChange={handleLabelSelect}
          >
            {labelOptions.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}

      <Typography variant="body2" color="text.secondary">
        Export a CSV where each row corresponds to a cell and contains its pixel
        intensities for the selected label and channel.
      </Typography>

      <Box>
        <Button
          variant="contained"
          onClick={handleExport}
          disabled={isExporting}
          startIcon={
            isExporting ? (
              <CircularProgress size={16} color="inherit" />
            ) : (
              <DownloadIcon />
            )
          }
        >
          {isExporting ? "Exporting..." : "Export"}
        </Button>
      </Box>
    </Box>
  );
};

export default PixelEngine;
