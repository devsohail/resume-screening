/**
 * Dashboard Component - Main overview page
 */

import React, { useEffect, useState } from 'react';
import { Grid, Paper, Typography, Box, CircularProgress } from '@mui/material';
import { apiClient } from '../services/api';

interface Metrics {
  total_jobs: number;
  total_resumes: number;
  total_screenings: number;
  shortlist_rate: number;
  average_score: number;
  average_processing_time_ms: number;
}

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const data = await apiClient.getAnalyticsMetrics();
      setMetrics(data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography color="textSecondary" gutterBottom>
              Total Jobs
            </Typography>
            <Typography variant="h3">{metrics?.total_jobs || 0}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography color="textSecondary" gutterBottom>
              Total Resumes
            </Typography>
            <Typography variant="h3">{metrics?.total_resumes || 0}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography color="textSecondary" gutterBottom>
              Screenings
            </Typography>
            <Typography variant="h3">{metrics?.total_screenings || 0}</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography color="textSecondary" gutterBottom>
              Shortlist Rate
            </Typography>
            <Typography variant="h3">{metrics?.shortlist_rate.toFixed(1) || 0}%</Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;

