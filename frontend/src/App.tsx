/**
 * Main App Component
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, Box } from '@mui/material';
import Dashboard from './components/Dashboard';
import JobManagement from './components/JobManagement';
import ScreeningView from './components/ScreeningView';
import Analytics from './components/Analytics';
import Navigation from './components/Navigation';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <Navigation />
          <Box component="main" sx={{ flexGrow: 1, p: 3, mt: 8 }}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/jobs" element={<JobManagement />} />
              <Route path="/screening/:jobId" element={<ScreeningView />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App;

