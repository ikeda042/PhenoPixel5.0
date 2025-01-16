// src/pages/TimelapseParser.tsx

import React, { useState } from "react";
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
  useTheme,
  TextField
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

// DisplayType を "raw" | "extracted" | "dual" に拡張
type DisplayType = "raw" | "extracted" | "dual";

const TimelapseParser: React.FC = () => {
  const [searchParams] = useSearchParams();
  const fileName = searchParams.get("file_name") || "";
  const [isLoading, setIsLoading] = useState(false);

  // パース完了状態を管理するステート
  const [isParsed, setIsParsed] = useState(false);

  // Field 関連 (GIF 表示のため)
  const [fields, setFields] = useState<string[]>([]);
  const [selectedField, setSelectedField] = useState<string>("");

  // param1 を文字列で管理
  const [param1, setParam1] = useState<string>("");

  // cropSize を文字列で管理
  const [cropSize, setCropSize] = useState<string>("200");

  // GIF 表示関連 (raw, extracted それぞれの URL)
  const [gifUrlRaw, setGifUrlRaw] = useState<string>("");
  const [gifUrlExtracted, setGifUrlExtracted] = useState<string>("");

  // 画像表示形式 (raw / extracted / dual)
  const [displayType, setDisplayType] = useState<DisplayType>("raw");

  // レスポンシブブレイクポイントを取得
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));

  /**
   * 指定した Field の GIF を取得 (displayType に応じて取得先を切り替え)
   */
  const fetchGif = async (fieldValue: string, type: DisplayType) => {
    if (!fileName || !fieldValue) return;
    setIsLoading(true);

    // Raw 用＆Extracted 用のエンドポイント
    const endpointRaw = `${url_prefix}/tlengine/nd2_files/${fileName}/gif/${fieldValue}`;
    // フィールドごとのエンドポイントは使わず、常に /cells/{fieldValue}/gif を使用
    const endpointExtracted = `${url_prefix}/tlengine/nd2_files/${fileName}/cells/${fieldValue}/gif`;

    try {
      if (type === "dual") {
        // 同時取得
        const [rawResponse, extractedResponse] = await Promise.all([
          axios.get(endpointRaw, { responseType: "blob" }),
          axios.get(endpointExtracted, { responseType: "blob" })
        ]);
        setGifUrlRaw(URL.createObjectURL(rawResponse.data));
        setGifUrlExtracted(URL.createObjectURL(extractedResponse.data));
      } else if (type === "raw") {
        // Raw だけ取得
        const rawResponse = await axios.get(endpointRaw, { responseType: "blob" });
        setGifUrlRaw(URL.createObjectURL(rawResponse.data));
        setGifUrlExtracted(""); // 他方はクリア
      } else {
        // Extracted だけ取得
        const extractedResponse = await axios.get(endpointExtracted, { responseType: "blob" });
        setGifUrlExtracted(URL.createObjectURL(extractedResponse.data));
        setGifUrlRaw(""); // 他方はクリア
      }
    } catch (error) {
      console.error("Failed to fetch GIF", error);
      // 取得に失敗した場合はそれぞれ空にしておく
      setGifUrlRaw("");
      setGifUrlExtracted("");
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * ND2 ファイルをパースする
   * パース完了時に Field 一覧を取得し、初期表示として "Field_1" の "raw" を自動表示
   */
  const handleParseND2 = async () => {
    if (!fileName) return;
    setIsLoading(true);
    try {
      // 1) ND2 ファイルの解析
      await axios.get(`${url_prefix}/tlengine/nd2_files/${fileName}`);

      // 2) Field 一覧を取得 (GIF 表示用)
      const fieldsResponse = await axios.get(
        `${url_prefix}/tlengine/nd2_files/${fileName}/fields`
      );

      if (fieldsResponse.data.fields) {
        setFields(fieldsResponse.data.fields);
      } else {
        setFields([]);
      }

      // パース完了
      setIsParsed(true);

      // パース完了時に "Field_1" & "raw" を自動取得
      setSelectedField("Field_1");
      setDisplayType("raw");
      await fetchGif("Field_1", "raw");
    } catch (error) {
      console.error("Failed to parse ND2 file or get fields", error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * "Extract cells"ボタン押下時に呼び出す
   * 全ての Field を対象にセルを抽出するが、param1, crop_size をクエリパラメータで指定
   * GET /tlengine/nd2_files/{file_name}/cells?param_1=...&crop_size=...
   */
  const handleExtractAllCells = async () => {
    if (!fileName) return;
    setIsLoading(true);
    try {
      await axios.get(`${url_prefix}/tlengine/nd2_files/${fileName}/cells`, {
        params: { param_1: Number(param1), crop_size: Number(cropSize) }
      });
      alert(
        `Cells have been extracted successfully! (param1=${param1}, crop_size=${cropSize})`
      );
    } catch (error) {
      console.error("Failed to extract cells", error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Field セレクト変更時のハンドラ (GIF 表示用)
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
  const handleDisplayTypeChange = (event: SelectChangeEvent<DisplayType>) => {
    const newType = event.target.value as DisplayType;
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

                {/* ND2 パースボタン (常時表示) */}
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

                {/* 以下の要素は isParsed = true の時だけ表示 */}
                {isParsed && (
                  <>
                    {/* param1入力フォーム */}
                    <TextField
                      label="param1"
                      type="text"
                      inputMode="numeric"
                      value={param1}
                      onChange={(e) => setParam1(e.target.value)}
                      fullWidth
                      sx={{ mb: 2 }}
                    />

                    {/* cropSize入力フォーム */}
                    <TextField
                      label="cropSize"
                      type="text"
                      inputMode="numeric"
                      value={cropSize}
                      onChange={(e) => setCropSize(e.target.value)}
                      fullWidth
                      sx={{ mb: 2 }}
                    />

                    {/* Extract cells ボタン */}
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

                    {/* Field ドロップダウン */}
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

                    {/* 表示形式ドロップダウン："raw" / "extracted" / "dual" */}
                    {fields.length > 0 && (
                      <FormControl fullWidth>
                        <InputLabel>Display Type</InputLabel>
                        <Select value={displayType} onChange={handleDisplayTypeChange}>
                          <MenuItem value="raw">Raw</MenuItem>
                          <MenuItem value="extracted">Extracted</MenuItem>
                          <MenuItem value="dual">Dual</MenuItem>
                        </Select>
                      </FormControl>
                    )}
                  </>
                )}
              </Grid>

              {/* 右サイド：GIF 表示エリア */}
              <Grid item xs={12} md={8}>
                {/* ディスプレイモードに応じて描画を切り替え */}
                {displayType === "raw" && gifUrlRaw && (
                  <Box
                    display="flex"
                    flexDirection="column"
                    alignItems="center"
                    sx={{ mt: isSmallScreen ? 2 : 0 }}
                  >
                    <Typography variant="body1" mb={2}>
                      <b>{selectedField}</b> (Raw)
                    </Typography>
                    <Box
                      component="img"
                      src={gifUrlRaw}
                      alt={`GIF (Raw) for Field: ${selectedField}`}
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
                )}

                {displayType === "extracted" && gifUrlExtracted && (
                  <Box
                    display="flex"
                    flexDirection="column"
                    alignItems="center"
                    sx={{ mt: isSmallScreen ? 2 : 0 }}
                  >
                    <Typography variant="body1" mb={2}>
                      <b>{selectedField}</b> (Extracted)
                    </Typography>
                    <Box
                      component="img"
                      src={gifUrlExtracted}
                      alt={`GIF (Extracted) for Field: ${selectedField}`}
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
                )}

                {displayType === "dual" && (
                  <Box
                    display="flex"
                    flexDirection={isSmallScreen ? "column" : "row"}
                    justifyContent="center"
                    alignItems="center"
                    gap={2}
                    sx={{ mt: isSmallScreen ? 2 : 0 }}
                  >
                    {/* Raw GIF */}
                    {gifUrlRaw && (
                      <Box
                        display="flex"
                        flexDirection="column"
                        alignItems="center"
                      >
                        <Typography variant="body1" mb={1}>
                          <b>{selectedField}</b> (Raw)
                        </Typography>
                        <Box
                          component="img"
                          src={gifUrlRaw}
                          alt={`GIF (Raw) for Field: ${selectedField}`}
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
                    )}
                    {/* Extracted GIF */}
                    {gifUrlExtracted && (
                      <Box
                        display="flex"
                        flexDirection="column"
                        alignItems="center"
                      >
                        <Typography variant="body1" mb={1}>
                          <b>{selectedField}</b> (Extracted)
                        </Typography>
                        <Box
                          component="img"
                          src={gifUrlExtracted}
                          alt={`GIF (Extracted) for Field: ${selectedField}`}
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
                    )}
                  </Box>
                )}

                {/* 何も選択されていないか、まだ GIF がない場合 */}
                {!gifUrlRaw && !gifUrlExtracted && (
                  <Box
                    display="flex"
                    justifyContent="center"
                    alignItems="center"
                    sx={{ height: "100%", minHeight: 200 }}
                  >
                    <Typography variant="body2" color="text.secondary">
                      Please parse ND2 file and select a field & display type
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
