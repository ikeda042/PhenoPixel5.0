
import './App.css';
import Nav from './components/Nav';
import { Box } from '@mui/system';
import SquareImage from './components/Squareimage';
import DBtable from './components/Dbtable';
import Dbcontents from './pages/Ddcontents';
import Grid from '@mui/material/Unstable_Grid2';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';  // Import Routes
import React, { useEffect, useState } from 'react';
import Cell from './pages/Cell'
import DbcontentsOverview from './pages/Cellsoverview';
import { settings } from './settings';
import MCPR from './pages/MCPR';
import Slots from './components/Slots';

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch(`${settings.url_prefix}/cellapi/cells/databases`)
      .then(response => response.json())
      .then(data => setData(data));
  }, []);

  return (
    <Router>
      <Box sx={{ bgcolor: "#f7f6f5", color: 'black', minHeight: '100vh' }}>

        <Routes>

          <Route path="/" element={
            <>
              <Nav title='データベース一括管理システム' />
              <Grid container spacing={4} margin={5}>
                <DBtable data={data} />
              </Grid >
            </>
          } />

          <Route path="/dbcontents/:filename" element={
            <>
              <Nav title='データベース一括管理システム' />
              <Dbcontents />
            </>} />
          <Route path="/dbcontents/:filename/cell/:cellId" element={
            <>
              <Nav title='データベース一括管理システム' />
              <Cell />
            </>} />
          <Route path="/dbcontents/:filename/overview" element={
            <>
              <Nav title='データベース一括管理システム' />
              <DbcontentsOverview />
            </>
          } />
          <Route path="/mcpr" element=
            {<>
              <Nav title='増殖曲線自動プロット' />
              <MCPR />
            </>
            } />
          <Route path="/slots" element=
            {<>
              <Nav title='顕微鏡予約状況' />
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
                <Slots />
              </Box>
            </>
            } />

        </Routes>


      </Box >
    </Router>
  );
}

export default App;