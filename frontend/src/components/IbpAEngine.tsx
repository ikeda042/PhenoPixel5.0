import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  Box,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { settings } from "../settings";

interface IbpAEngineProps {
  dbName: string;
  label: string;
  cellId: string;
}

const url_prefix = settings.url_prefix;

const IbpAEngine: React.FC<IbpAEngineProps> = ({ dbName, label, cellId }) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedChannel, setSelectedChannel] = useState<"fluo1" | "fluo2">(
    "fluo1"
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    let objectUrl: string | null = null;

    const fetchBoxPlot = async () => {
      setLoading(true);
      try {
        setError(null);
        setImageUrl(null);
        const response = await axios.get(
          `${url_prefix}/cells/${dbName}/${label}/${cellId}/ibpa_ratio`,
          {
            responseType: "blob",
            params: { img_type: selectedChannel },
          }
        );
        if (!isMounted) {
          return;
        }
        objectUrl = URL.createObjectURL(response.data);
        setImageUrl(objectUrl);
      } catch (err) {
        console.error("Failed to fetch IbpA ratio plot:", err);
        if (isMounted) {
          setImageUrl(null);
          setError("No plot available for the selected channel.");
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchBoxPlot();

    return () => {
      isMounted = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [dbName, label, cellId, selectedChannel]);

  const handleChannelChange = (
    _event: React.MouseEvent<HTMLElement>,
    newChannel: "fluo1" | "fluo2" | null
  ) => {
    if (newChannel) {
      setSelectedChannel(newChannel);
    }
  };

  return (
    <Box display="flex" flexDirection="column" alignItems="flex-end">
      <Box display="flex" flexDirection="column" alignItems="flex-end" mb={2}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Fluorescence channel
        </Typography>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={selectedChannel}
          onChange={handleChannelChange}
        >
          <ToggleButton value="fluo1">Fluo 1</ToggleButton>
          <ToggleButton value="fluo2">Fluo 2</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      {loading ? (
        <Box display="flex" justifyContent="center" width="100%">
          <CircularProgress />
        </Box>
      ) : imageUrl ? (
        <img
          src={imageUrl}
          alt="IbpA ratio box plot"
          style={{ maxWidth: "100%", height: "auto" }}
        />
      ) : (
        <Box>{error ?? "No plot available"}</Box>
      )}
    </Box>
  );
};

export default IbpAEngine;
