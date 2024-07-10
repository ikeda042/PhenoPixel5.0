
import './App.css';
import Nav from './components/Nav';
import { Box } from '@mui/system';
import Grid from '@mui/material/Unstable_Grid2';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import React, { useEffect, useState } from 'react';
import CellImageGrid from './components/CellOverview';
import TopPage from './components/TopPage';

function App() {


  return (
    <Router>
      <Box sx={{ bgcolor: "#f7f6f5", color: 'black', minHeight: '100vh' }}>

        <Routes>
          <Route path="/" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={4} margin={5}>
                <TopPage />
              </Grid >
            </>
          } />
          <Route path="/databases/" element={
            <>
              <Nav title='PhenoPixel5.0' />
              <Grid container spacing={4} margin={5}>
                <CellImageGrid />
              </Grid >
            </>
          } />





        </Routes>


      </Box >
    </Router>
  );
}

export default App;