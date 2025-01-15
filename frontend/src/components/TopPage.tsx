import React, { useEffect, useState, useCallback } from "react";
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Container,
  Switch,
  FormControlLabel,
} from "@mui/material";
import DatabaseIcon from "@mui/icons-material/Storage";
import ScienceIcon from "@mui/icons-material/Science";
import TerminalIcon from "@mui/icons-material/Terminal";
import GitHubIcon from "@mui/icons-material/GitHub";
import BarChartIcon from "@mui/icons-material/BarChart";
import DisplaySettingsIcon from "@mui/icons-material/DisplaySettings";
import Inventory2Icon from "@mui/icons-material/Inventory2";
import SettingsEthernetIcon from "@mui/icons-material/SettingsEthernet";
import { useNavigate } from "react-router-dom";
import { settings } from "../settings";
import AutoAwesomeMotionIcon from '@mui/icons-material/AutoAwesomeMotion';

/* --------------------------------
 *  ImageCard Component
 * --------------------------------*/
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
        height: { xs: 200, md: 220 },
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        boxShadow: 6,
        transition: "box-shadow 0.3s ease-in-out",
        "&:hover": {
          backgroundColor: "lightgrey",
          boxShadow: 10,
        },
      }}
    >
      {imageUrl && (
        <Box
          component="img"
          src={imageUrl}
          alt={`${title} Image`}
          sx={{
            height: { xs: 100, md: 120 },
            width: "100%",
            objectFit: "cover",
          }}
        />
      )}
      <CardContent
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          "&:last-child": { pb: 2 },
        }}
      >
        <Box
          sx={{
            marginTop: 2,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            width: "100%",
          }}
        >
          <Typography variant="h6" noWrap>
            {title}
          </Typography>
        </Box>
        <Typography
          variant="body2"
          color="textSecondary"
          noWrap
          sx={{
            textAlign: "center",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            width: "100%",
          }}
        >
          {description}
        </Typography>
      </CardContent>
    </Card>
  );
};

/* --------------------------------
 *  Main Page Component
 * --------------------------------*/

const cell_ids = [
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

// ランダムに cellId を抽出
const cellId = cell_ids[Math.floor(Math.random() * cell_ids.length)];

const TopPage: React.FC = () => {
  const navigate = useNavigate();

  // ステータス類
  const [backendStatus, setBackendStatus] = useState<string | null>(null);
  const [dropboxStatus, setDropboxStatus] = useState<boolean | null>(null);
  const [internetStatus, setInternetStatus] = useState<boolean | null>(null);

  // デモ表示スイッチ
  const [showDemo, setShowDemo] = useState<boolean>(false);

  // 画像URLを保持
  const [image3DUrl1, setImage3DUrl1] = useState<string | null>(null);
  const [image3DUrl2, setImage3DUrl2] = useState<string | null>(null);
  const [image3DUrl3, setImage3DUrl3] = useState<string | null>(null);
  const [image3DUrl4, setImage3DUrl4] = useState<string | null>(null);

  /**
   * Backend, Dropbox, Internet 接続ステータスをチェック
   */
  const checkStatuses = useCallback(async () => {
    try {
      // Backend
      const backendResponse = await fetch(`${settings.url_prefix}/healthcheck`);
      if (backendResponse.status === 200) {
        setBackendStatus("ready");
      } else {
        setBackendStatus("not working");
      }

      // Dropbox
      const dropboxRes = await fetch(
        `${settings.url_prefix}/dropbox/connection_check`
      );
      const dropboxData = await dropboxRes.json();
      if (dropboxRes.status === 200 && dropboxData.status) {
        setDropboxStatus(true);
      } else {
        setDropboxStatus(false);
      }

      // Internet
      const internetRes = await fetch(`${settings.url_prefix}/internet-connection`);
      const internetData = await internetRes.json();
      if (internetRes.status === 200 && internetData.status) {
        setInternetStatus(true);
      } else {
        setInternetStatus(false);
      }
    } catch (error) {
      console.error("Error checking statuses:", error);
      setBackendStatus("not working");
      setDropboxStatus(false);
      setInternetStatus(false);
    }
  }, []);

  useEffect(() => {
    checkStatuses();
  }, [checkStatuses]);

  /**
   * Demoデータセット（画像URL）を取得
   */
  useEffect(() => {
    // スイッチOFF の場合はリセットして終了
    if (!showDemo) {
      setImage3DUrl1(null);
      setImage3DUrl2(null);
      setImage3DUrl3(null);
      setImage3DUrl4(null);
      return;
    }

    // スイッチON の場合は画像取得処理
    const fetchDemoImages = async () => {
      try {
        const [res1, res2, res3, res4] = await Promise.all([
          fetch(`${settings.url_prefix}/cells/${cellId}/test_database.db/false/false/ph_image`),
          fetch(`${settings.url_prefix}/cells/${cellId}/test_database.db/false/false/fluo_image`),
          fetch(`${settings.url_prefix}/cells/${cellId}/test_database.db/replot?degree=3`),
          fetch(`${settings.url_prefix}/cells/test_database.db/${cellId}/3d`),
        ]);

        // PH
        if (res1.ok) {
          const blob1 = await res1.blob();
          setImage3DUrl1(URL.createObjectURL(blob1));
        }

        // Fluo
        if (res2.ok) {
          const blob2 = await res2.blob();
          setImage3DUrl2(URL.createObjectURL(blob2));
        }

        // Replot
        if (res3.ok) {
          const blob3 = await res3.blob();
          setImage3DUrl3(URL.createObjectURL(blob3));
        }

        // 3D
        if (res4.ok) {
          const blob4 = await res4.blob();
          setImage3DUrl4(URL.createObjectURL(blob4));
        }
      } catch (error) {
        console.error("Error fetching demo images:", error);
      }
    };

    fetchDemoImages();

    // クリーンアップ関数でURLオブジェクトを解放
    return () => {
      if (image3DUrl1) URL.revokeObjectURL(image3DUrl1);
      if (image3DUrl2) URL.revokeObjectURL(image3DUrl2);
      if (image3DUrl3) URL.revokeObjectURL(image3DUrl3);
      if (image3DUrl4) URL.revokeObjectURL(image3DUrl4);
    };
  }, [showDemo, cellId, image3DUrl1, image3DUrl2, image3DUrl3, image3DUrl4]);

  /**
   * ページ遷移ハンドラ
   */
  const handleNavigate = (path: string, external?: boolean) => {
    if (external) {
      window.open(path, "_blank");
    } else {
      navigate(path);
    }
  };

  /**
   * Demo表示スイッチ ON/OFF
   */
  const handleToggle = (event: React.ChangeEvent<HTMLInputElement>) => {
    setShowDemo(event.target.checked);
  };

  /**
   * メニューアイテム
   * （System Status のメニュー項目は削除し、代わりに上部に表示）
   */
  const menuItems = [
    {
      title: "Database Console",
      icon: <DatabaseIcon sx={{ fontSize: 50 }} />,
      path: "/dbconsole",
      description: "Label cells / manage databases.",
      external: false,
    },
    {
      title: "Cell Extraction",
      icon: <ScienceIcon sx={{ fontSize: 50 }} />,
      path: "/nd2files",
      description: "Extract cells from ND2 files.",
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
      title: "TimeLapse Engine",
      icon: <DisplaySettingsIcon sx={{ fontSize: 50 }} />,
      path: "/tl-engine",
      description: "Process nd2 timelapse files.(beta)",
      external: false,
    },
    {
      title: "Timelapse Database",
      icon: <AutoAwesomeMotionIcon sx={{ fontSize: 50 }} />,
      path: "/tl-engine",
      description: "Timelapse database console (beta)",
      external: false,
    },
    {
      title: "Swagger UI",
      icon: <TerminalIcon sx={{ fontSize: 50 }} />,
      path: `${settings.url_prefix}/docs`,
      description: "Test the API endpoints.",
      external: true,
    },
    {
      title: "Github",
      icon: <GitHubIcon sx={{ fontSize: 50 }} />,
      path: "https://github.com/ikeda042/PhenoPixel5.0",
      description: "Project documentation.",
      external: true,
    },
  ];

  return (
    <Container sx={{ minHeight: "100vh", py: 4 }}>
      {/* 上部に System Status を表示させる */}
      <Box sx={{ mb: 4 }}>
        <Box
          sx={{
            border: "1px solid grey",
            borderRadius: "8px",
            padding: 2,
            textAlign: "center",
          }}
        >
          <Typography variant="h6" sx={{ mb: 1 }}>
            System Status
          </Typography>
          <Typography
            variant="body2"
            color={backendStatus === "ready" ? "green" : "red"}
          >
            Backend: {backendStatus || "unknown"} ({settings.url_prefix})
          </Typography>
          <Typography
            variant="body2"
            color={dropboxStatus ? "green" : "red"}
          >
            Dropbox:{" "}
            {dropboxStatus !== null
              ? dropboxStatus
                ? "Connected"
                : "Not connected"
              : "unknown"}
          </Typography>
          <Typography
            variant="body2"
            color={internetStatus ? "green" : "red"}
          >
            Internet:{" "}
            {internetStatus !== null
              ? internetStatus
                ? "Connected"
                : "Not connected"
              : "unknown"}
          </Typography>
        </Box>
      </Box>

      <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center">
        <Grid container spacing={2} justifyContent="center">
          {/* スイッチの追加 */}
          <Grid item xs={12} sx={{ textAlign: "center", mb: 2 }}>
            <FormControlLabel
              control={<Switch checked={showDemo} onChange={handleToggle} color="success" />}
              label="Show Demo Dataset"
            />
          </Grid>

          {/* Demo dataset の条件付きレンダリング */}
          {showDemo && (
            <>
              <Grid item xs={12}>
                <Typography variant="h5" mb={2} textAlign="center">
                  Demo dataset for <b>{cellId}</b>
                </Typography>
              </Grid>

              <Grid item xs={12} sm={6} md={3}>
                <ImageCard
                  title="PH"
                  description={`PH image of ${cellId}.`}
                  imageUrl={image3DUrl1}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <ImageCard
                  title="Fluo"
                  description={`Fluo image of ${cellId}.`}
                  imageUrl={image3DUrl2}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <ImageCard
                  title="Replotted"
                  description={`Replotted image of ${cellId}.`}
                  imageUrl={image3DUrl3}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <ImageCard
                  title="3D Plot"
                  description={`3D plot of ${cellId}.`}
                  imageUrl={image3DUrl4}
                />
              </Grid>
            </>
          )}

          {/* メインメニューをカードで表示 */}
          {menuItems.map((item, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card
                onClick={() => handleNavigate(item.path, item.external)}
                sx={{
                  cursor: "pointer",
                  textAlign: "center",
                  height: { xs: 180, md: 200 },
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  boxShadow: 6,
                  transition: "box-shadow 0.3s ease-in-out",
                  "&:hover": {
                    backgroundColor: "lightgrey",
                    boxShadow: 10,
                  },
                }}
              >
                <CardContent
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    "&:last-child": { pb: 2 },
                  }}
                >
                  {item.icon}
                  <Typography variant="h6" mt={2}>
                    {item.title}
                  </Typography>
                  {typeof item.description === "string" ? (
                    <Typography variant="body2" mt={1} color="textSecondary">
                      {item.description}
                    </Typography>
                  ) : (
                    <Box mt={1}>{item.description}</Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Container>
  );
};

export default TopPage;
