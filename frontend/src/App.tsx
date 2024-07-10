
import './App.css';
import Nav from './components/Nav';
import { Box } from '@mui/system';
// import SquareImage from './components/Squareimage';
// import DBtable from './components/Dbtable';
// import Dbcontents from './pages/Ddcontents';
import Grid from '@mui/material/Unstable_Grid2';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import React, { useEffect, useState } from 'react';
// import Cell from './pages/Cell'
// import DbcontentsOverview from './pages/Cellsoverview';
// import { settings } from './settings';
// import MCPR from './pages/MCPR';
// import Slots from './components/Slots';
import { settings } from './settings';
import CellImageGrid from './components/CellOverview';


function App() {
  // const [data, setData] = useState([]);

  // useEffect(() => {
  //   fetch(`${settings.url_prefix}/cellapi/cells/databases`)
  //     .then(response => response.json())
  //     .then(data => setData(data));
  // }, []);

  return (
    <Router>
      <Box sx={{ bgcolor: "#f7f6f5", color: 'black', minHeight: '100vh' }}>

        <Routes>

          <Route path="/databases" element={
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