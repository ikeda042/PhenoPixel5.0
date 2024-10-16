
import './App.css';
import Nav from './components/Nav';
import { Box } from '@mui/system';
import Grid from '@mui/material/Unstable_Grid2';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import CellImageGrid from './components/CellOverview';
import Databases from './components/Databases';
import TopPage from './components/TopPage';
import Nd2Files from './components/Nd2files';
import CellExtraction from './components/CellExtraction';
import GraphEngine from './components/GraphEngine';
import TimelapseNd2List from './components/TimelapseNd2List';
import TimelapseParser from './components/TimelapseParser';
import ResultsConsole from './components/ResultsConsole';

function App() {
  return (
    <Box sx={{ bgcolor: "000", color: 'black', minHeight: '100vh' }}>
      <Router>
        <Routes>
          <Route path="/" element={
            <>
              <Nav title={`PhenoPixel5.0`} />
              <Grid container spacing={4} margin={5} mt={-20}>
                <TopPage />
              </Grid >
            </>
          } />
          <Route path="/dbconsole" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={4} margin={5} mt={-4}>
                <Databases />
              </Grid >
            </>
          } />
          <Route path="/databases" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={1} margin={3} mt={-4}>
                <CellImageGrid />
              </Grid >
            </>
          } />
          <Route path="/nd2files" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={1} margin={3} mt={-4}>
                <Nd2Files />
              </Grid >
            </>
          } />
          <Route path="/cellextraction" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={1} margin={3} mt={-4}>
                <CellExtraction />
              </Grid >
            </>
          } />
          <Route path="/graphengine" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={1} margin={3} mt={-4}>
                <GraphEngine />
              </Grid >
            </>
          } />
          <Route path="/tl-engine" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={1} margin={3} mt={-4}>
                <TimelapseNd2List />
              </Grid >
            </>
          } />
          <Route path="/tlparser" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={1} margin={3} mt={-4}>
                <TimelapseParser />
              </Grid >
            </>
          } />
          <Route path="/results" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={1} margin={3} mt={-4}>
                <ResultsConsole />
              </Grid >
            </>
          } />
        </Routes>
      </Router>
    </Box>
  );
}

export default App;