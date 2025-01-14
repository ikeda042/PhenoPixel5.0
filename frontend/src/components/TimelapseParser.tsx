import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Box,
  Grid,
  Typography,
  Button,
  Backdrop,
  CircularProgress,
  Breadcrumbs,
  Link,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Card,
  CardContent,
  useMediaQuery,
  useTheme
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const TimelapseParser: React.FC = () => {
  const [searchParams] = useSearchParams();
  const fileName = searchParams.get("file_name") || "";
  const [isLoading, setIsLoading] = useState(false);

  // Field 関連
  const [fields, setFields] = useState<string[]>([]);
  const [selectedField, setSelectedField] = useState<string>("");

  // GIF 表示関連
  const [gifUrl, setGifUrl] = useState<string>("");

  /**
   * 画像表示形式 (生 / 細胞抽出後) を管理
   *  - raw: ParseND2File 処理で生成した生の画像 ( /nd2_files/{file_name}/gif/{field_name} )
   *  - extracted: ExtractCells 後の画像 ( /nd2_files/{file_name}/cells/{field_name}/gif )
   */
  const [displayType, setDisplayType] = useState<"raw" | "extracted">("raw");

  // レスポンシブブレイクポイントを取得
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  /**
   * ND2 ファイルをパースする
   * 1) GET /tlengine/nd2_files/{file_name}
   * 2) パース完了後に、GET /tlengine/nd2_files/{file_name}/fields を呼び出す
   */
  const handleParseND2 = async () => {
    if (!fileName) return;
    setIsLoading(true);
    try {
      // 1) ND2 ファイルの解析
      await axios.get(`${url_prefix}/tlengine/nd2_files/${fileName}`);
      // 2) Field 一覧を取得
      const fieldsResponse = await axios.get(
        `${url_prefix}/tlengine/nd2_files/${fileName}/fields`
      );
      if (fieldsResponse.data.fields) {
        setFields(fieldsResponse.data.fields);
      } else {
        setFields([]);
      }
    } catch (error) {
      console.error("Failed to parse ND2 file or get fields", error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * "Extract cells"ボタン押下時に呼び出す
   * GET /tlengine/nd2_files/{file_name}/cells
   */
  const handleExtractAllCells = async () => {
    if (!fileName) return;
    setIsLoading(true);
    try {
      await axios.get(`${url_prefix}/tlengine/nd2_files/${fileName}/cells`);
      alert("Cells have been extracted successfully!");
    } catch (error) {
      console.error("Failed to extract cells", error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Field 選択時 (または表示形式選択時) に呼び出す
   * - 生画像:     GET /tlengine/nd2_files/{file_name}/gif/{field_name}
   * - 抽出後画像: GET /tlengine/nd2_files/{file_name}/cells/{field_name}/gif
   */
  const fetchGif = async (fieldValue: string, type: "raw" | "extracted") => {
    if (!fileName || !fieldValue) return;
    setIsLoading(true);

    // エンドポイントを動的に切り替え
    const endpoint =
      type === "raw"
        ? `${url_prefix}/tlengine/nd2_files/${fileName}/gif/${fieldValue}`
        : `${url_prefix}/tlengine/nd2_files/${fileName}/cells/${fieldValue}/gif`;

    try {
      const response = await axios.get(endpoint, { responseType: "blob" });
      const blobUrl = URL.createObjectURL(response.data);
      setGifUrl(blobUrl);
    } catch (error) {
      console.error("Failed to fetch GIF", error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Field セレクト変更時のハンドラ
   */
  const handleFieldChange = (event: SelectChangeEvent<string>) => {
    const newField = event.target.value as string;
    setSelectedField(newField);

    // Field を選択する度に現在の displayType で画像を取得し直す
    if (newField) {
      fetchGif(newField, displayType);
    }
  };

  /**
   * 表示形式ドロップダウン変更時のハンドラ
   */
  const handleDisplayTypeChange = (event: SelectChangeEvent<"raw" | "extracted">) => {
    const newType = event.target.value as "raw" | "extracted";
    setDisplayType(newType);

    // 既に Field が選択されている場合は再度取得
    if (selectedField) {
      fetchGif(selectedField, newType);
    }
  };

  return (
    <>
      <Backdrop open={isLoading} sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <CircularProgress color="inherit" />
      </Backdrop>

      {/* 全画面を覆うレイアウトコンテナ */}
      <Box
        sx={{
          width: "100%",
          minHeight: "100vh",
          boxSizing: "border-box",
          p: 4
        }}
      >
        {/* パンくずリスト */}
        <Box mb={2}>
          <Breadcrumbs aria-label="breadcrumb">
            <Link underline="hover" color="inherit" href="/">
              Top
            </Link>
            <Link underline="hover" color="inherit" href="/tl-engine">
              Timelapse ND2 files
            </Link>
            <Typography color="text.primary">ND2 parse</Typography>
          </Breadcrumbs>
        </Box>

        <Card sx={{ height: "100%" }}>
          <CardContent>
            <Grid container spacing={3} sx={{ height: "100%" }}>
              {/* 左サイド：ファイル名と操作ボタン */}
              <Grid item xs={12} md={4}>
                <Typography
                  variant={isSmallScreen ? "body1" : "h6"}
                  mb={2}
                  sx={{ fontWeight: 600 }}
                >
                  ND2 filename: {fileName}
                </Typography>

                {/* ND2 パースボタン */}
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  onClick={handleParseND2}
                  disabled={isLoading || !fileName}
                  sx={{
                    height: 56,
                    mb: 2,
                    textTransform: "none",
                    backgroundColor: "#333",
                    "&:hover": {
                      backgroundColor: "#555"
                    }
                  }}
                >
                  Parse ND2 File
                </Button>

                {/* Extract cells ボタン：fields が取得できたら表示 */}
                {fields.length > 0 && (
                  <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    onClick={handleExtractAllCells}
                    disabled={isLoading || !fileName}
                    sx={{
                      height: 56,
                      mb: 2,
                      textTransform: "none",
                      backgroundColor: "#333",
                      "&:hover": {
                        backgroundColor: "#555"
                      }
                    }}
                  >
                    Extract cells
                  </Button>
                )}

                {/* Field ドロップダウン：fields が取得できたら表示 */}
                {fields.length > 0 && (
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Field</InputLabel>
                    <Select value={selectedField} onChange={handleFieldChange}>
                      {fields.map((field) => (
                        <MenuItem key={field} value={field}>
                          {field}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}

                {/* 表示形式ドロップダウン："raw" / "extracted" */}
                {fields.length > 0 && (
                  <FormControl fullWidth>
                    <InputLabel>Display Type</InputLabel>
                    <Select value={displayType} onChange={handleDisplayTypeChange}>
                      <MenuItem value="raw">Raw</MenuItem>
                      <MenuItem value="extracted">Extracted</MenuItem>
                    </Select>
                  </FormControl>
                )}
              </Grid>

              {/* 右サイド：GIF 表示エリア */}
              <Grid item xs={12} md={8}>
                {gifUrl ? (
                  <Box
                    display="flex"
                    flexDirection="column"
                    alignItems="center"
                    sx={{ mt: isSmallScreen ? 2 : 0 }}
                  >
                    <Typography variant="body1" mb={2}>
                      <b>{selectedField}</b> ({displayType === "raw" ? "Raw" : "Extracted"})
                    </Typography>
                    <Box
                      component="img"
                      src={gifUrl}
                      alt={`GIF for Field: ${selectedField}`}
                      sx={{
                        maxWidth: "100%",
                        height: "auto",
                        objectFit: "contain",
                        border: "1px solid #eee",
                        borderRadius: 1,
                        p: 1
                      }}
                    />
                  </Box>
                ) : (
                  <Box
                    display="flex"
                    justifyContent="center"
                    alignItems="center"
                    sx={{ height: "100%", minHeight: 200 }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      Please wait for parsing ND2 file and select a field
                    </Typography>
                  </Box>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>
    </>
  );
};

export default TimelapseParser;
