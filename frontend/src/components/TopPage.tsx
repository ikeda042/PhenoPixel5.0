import React, { useEffect, useState, useCallback, useMemo } from "react";
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Container,
  Switch,
  FormControlLabel,
  Paper,
} from "@mui/material";
import DatabaseIcon from "@mui/icons-material/Storage";
import ScienceIcon from "@mui/icons-material/Science";
import TerminalIcon from "@mui/icons-material/Terminal";
import DriveFileMoveIcon from '@mui/icons-material/DriveFileMove';
import BarChartIcon from "@mui/icons-material/BarChart";
import DisplaySettingsIcon from "@mui/icons-material/DisplaySettings";
import Inventory2Icon from "@mui/icons-material/Inventory2";
import AutoAwesomeMotionIcon from "@mui/icons-material/AutoAwesomeMotion";
import { useNavigate } from "react-router-dom";
import { settings } from "../settings";

// ------------------------------
// ImageCard Component
// ------------------------------
interface ImageCardProps {
  title: string;
  description: string;
  imageUrl: string | null;
}

const ImageCard: React.FC<ImageCardProps> = ({ title, description, imageUrl }) => {
  return (
    <Card
      sx={{
        cursor: "pointer",
        textAlign: "center",
        height: { xs: 220, md: 240 },
        borderRadius: 2,
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        boxShadow: 3,
        transition: "transform 0.3s, box-shadow 0.3s",
        "&:hover": {
          transform: "scale(1.03)",
          boxShadow: 6,
          backgroundColor: "rgba(0,0,0,0.04)",
        },
      }}
    >
      {imageUrl && (
        <Box
          component="img"
          src={imageUrl}
          alt={`${title} Image`}
          sx={{
            height: { xs: 120, md: 140 },
            width: "100%",
            objectFit: "cover",
            borderTopLeftRadius: 8,
            borderTopRightRadius: 8,
          }}
        />
      )}
      <CardContent
        sx={{
          flex: 1,
          p: 2,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
      >
        <Typography variant="h6" noWrap sx={{ mb: 1 }}>
          {title}
        </Typography>
        <Typography
          variant="body2"
          color="text.secondary"
          noWrap
          sx={{ overflow: "hidden", textOverflow: "ellipsis" }}
        >
          {description}
        </Typography>
      </CardContent>
    </Card>
  );
};

// ------------------------------
// StatusBar Component
// ------------------------------
interface StatusBarProps {
  backendStatus: string | null;
  internetStatus: boolean | null;
}

const StatusBar: React.FC<StatusBarProps> = ({
  backendStatus,
  internetStatus,
}) => {
  return (
    <Paper elevation={3} sx={{ mb: 4, p: 2, borderRadius: 2 }}>
      <Typography variant="h6" sx={{ mb: 1, textAlign: "center" }}>
        System Status
      </Typography>
      <Box sx={{ display: "flex", justifyContent: "space-around" }}>
        <Typography
          variant="body2"
          color={backendStatus === "ready" ? "success.main" : "error.main"}
        >
          Backend: {backendStatus || "unknown"}
        </Typography>
        <Typography
          variant="body2"
          color={internetStatus ? "success.main" : "error.main"}
        >
          Internet:{" "}
          {internetStatus !== null ? (internetStatus ? "Connected" : "Not connected") : "unknown"}
        </Typography>
      </Box>
    </Paper>
  );
};

// ------------------------------
// DemoDataset Component
// ------------------------------
interface DemoDatasetProps {
  showDemo: boolean;
  cellId: string;
}

const DemoDataset: React.FC<DemoDatasetProps> = ({ showDemo, cellId }) => {
  const [imageUrls, setImageUrls] = useState<{
    ph: string | null;
    fluo: string | null;
    replot: string | null;
    plot3d: string | null;
  }>({
    ph: null,
    fluo: null,
    replot: null,
    plot3d: null,
  });

  useEffect(() => {
    if (!showDemo) {
      setImageUrls({ ph: null, fluo: null, replot: null, plot3d: null });
      return;
    }

    let active = true;
    const fetchDemoImages = async () => {
      try {
        const [res1, res2, res3, res4] = await Promise.all([
          fetch(`${settings.url_prefix}/cells/${cellId}/test_database.db/false/false/ph_image`),
          fetch(`${settings.url_prefix}/cells/${cellId}/test_database.db/false/false/fluo_image`),
          fetch(`${settings.url_prefix}/cells/${cellId}/test_database.db/replot?degree=3`),
          fetch(`${settings.url_prefix}/cells/test_database.db/${cellId}/3d`),
        ]);

        const newImageUrls = {
          ph: res1.ok ? URL.createObjectURL(await res1.blob()) : null,
          fluo: res2.ok ? URL.createObjectURL(await res2.blob()) : null,
          replot: res3.ok ? URL.createObjectURL(await res3.blob()) : null,
          plot3d: res4.ok ? URL.createObjectURL(await res4.blob()) : null,
        };

        if (active) {
          setImageUrls(newImageUrls);
        }
      } catch (error) {
        console.error("Error fetching demo images:", error);
      }
    };

    fetchDemoImages();

    return () => {
      active = false;
      Object.values(imageUrls).forEach((url) => {
        if (url) URL.revokeObjectURL(url);
      });
    };
  }, [showDemo, cellId]);

  if (!showDemo) return null;

  return (
    <>
      <Grid item xs={12}>
        <Typography variant="h5" sx={{ mb: 2, textAlign: "center" }}>
          Demo dataset for <b>{cellId}</b>
        </Typography>
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <ImageCard title="PH" description={`PH image of ${cellId}.`} imageUrl={imageUrls.ph} />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <ImageCard title="Fluo" description={`Fluo image of ${cellId}.`} imageUrl={imageUrls.fluo} />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <ImageCard title="Replotted" description={`Replotted image of ${cellId}.`} imageUrl={imageUrls.replot} />
      </Grid>
      <Grid item xs={12} sm={6} md={3}>
        <ImageCard title="3D Plot" description={`3D plot of ${cellId}.`} imageUrl={imageUrls.plot3d} />
      </Grid>
    </>
  );
};

// ------------------------------
// MenuGrid Component
// ------------------------------
interface MenuGridProps {
  handleNavigate: (path: string, external?: boolean) => void;
}

const MenuGrid: React.FC<MenuGridProps> = ({ handleNavigate }) => {
  const menuItems = useMemo(
    () => [
      {
        title: "Cell Extraction",
        icon: <ScienceIcon sx={{ fontSize: 50 }} />,
        path: "/nd2files",
        description: "Extract cells from ND2 files.",
        external: false,
      },
      {
        title: "Database Console",
        icon: <DatabaseIcon sx={{ fontSize: 50 }} />,
        path: "/dbconsole",
        description: "Label cells / manage databases.",
        external: false,
      },
      {
        title: "Results",
        icon: <Inventory2Icon sx={{ fontSize: 50 }} />,
        path: "/results",
        description: "Results for the queued jobs.",
        external: false,
      },
      {
        title: "GraphEngine",
        icon: <BarChartIcon sx={{ fontSize: 50 }} />,
        path: "/graphengine",
        description: "Create graphs from the data.",
        external: false,
      },
      {
        title: "TimeLapse Extraction",
        icon: <DisplaySettingsIcon sx={{ fontSize: 50 }} />,
        path: "/tl-engine",
        description: "Process nd2 timelapse files.(beta)",
        external: false,
      },
      {
        title: "Timelapse Database",
        icon: <AutoAwesomeMotionIcon sx={{ fontSize: 50 }} />,
        path: "/tlengine/dbconsole",
        description: "Timelapse database console (beta)",
        external: false,
      },
      {
        title: "CDT",
        icon: <TerminalIcon sx={{ fontSize: 50 }} />,
        path: "/cdt",
        description: "Calculate nagg from CSV files.",
        external: false,
      },
      {
        title: "File Manager",
        icon: <DriveFileMoveIcon sx={{ fontSize: 50 }} />,
        path: "/files",
        description: "Manage files on the local server.",
        external: true,
      },
    ],
    []
  );

  return (
    <>
      {menuItems.map((item, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card
            onClick={() => handleNavigate(item.path, item.external)}
            sx={{
              cursor: "pointer",
              textAlign: "center",
              height: { xs: 180, md: 200 },
              borderRadius: 2,
              display: "flex",
              flexDirection: "column",
              justifyContent: "center",
              boxShadow: 3,
              transition: "transform 0.3s, box-shadow 0.3s",
              "&:hover": {
                transform: "scale(1.03)",
                boxShadow: 6,
                backgroundColor: "rgba(0,0,0,0.04)",
              },
            }}
          >
            <CardContent sx={{ display: "flex", flexDirection: "column", alignItems: "center", p: 2 }}>
              {item.icon}
              <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>
                {item.title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {item.description}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </>
  );
};

// ------------------------------
// Main TopPage Component
// ------------------------------
const cellIds = [
  "F0C1",
  "F0C2",
  "F0C3",
  "F0C7",
  "F0C9",
  "F0C11",
  "F0C16",
  "F0C18",
  "F0C20",
  "F0C21",
  "F0C24",
  "F0C27",
  "F1C0",
  "F1C2",
  "F1C3",
  "F1C6",
  "F1C9",
  "F1C10",
  "F1C11",
  "F1C12",
  "F1C13",
  "F1C20",
];
const randomCellId = cellIds[Math.floor(Math.random() * cellIds.length)];

const TopPage: React.FC = () => {
  const navigate = useNavigate();

  const [backendStatus, setBackendStatus] = useState<string | null>(null);
  const [internetStatus, setInternetStatus] = useState<boolean | null>(null);
  const [showDemo, setShowDemo] = useState<boolean>(false);

  const checkStatuses = useCallback(async () => {
    try {
      const backendResponse = await fetch(`${settings.url_prefix}/healthcheck`);
      setBackendStatus(backendResponse.status === 200 ? "ready" : "not working");

      const internetRes = await fetch(`${settings.url_prefix}/internet-connection`);
      const internetData = await internetRes.json();
      setInternetStatus(internetRes.status === 200 && internetData.status);
    } catch (error) {
      console.error("Error checking statuses:", error);
      setBackendStatus("not working");
      setInternetStatus(false);
    }
  }, []);

  useEffect(() => {
    checkStatuses();
  }, [checkStatuses]);

  const handleNavigate = (path: string, external?: boolean) => {
    if (external) {
      window.open(path, "_blank");
    } else {
      navigate(path);
    }
  };

  const handleToggleDemo = (event: React.ChangeEvent<HTMLInputElement>) => {
    setShowDemo(event.target.checked);
  };

  return (
    <Container maxWidth="lg" sx={{ minHeight: "100vh", py: 4 }}>
      <StatusBar
        backendStatus={backendStatus}
        internetStatus={internetStatus}
      />
      <Box display="flex" flexDirection="column" alignItems="center">
        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} sx={{ textAlign: "center" }}>
            <FormControlLabel
              control={<Switch checked={showDemo} onChange={handleToggleDemo} color="success" />}
              label="Show Demo Dataset"
            />
          </Grid>
          <DemoDataset showDemo={showDemo} cellId={randomCellId} />
          <MenuGrid handleNavigate={handleNavigate} />
        </Grid>
      </Box>
    </Container>
  );
};

export default TopPage;
