/**
 * Screening View Component - View results for a job
 */

import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Chip,
  CircularProgress,
} from '@mui/material';
import { apiClient } from '../services/api';

const ScreeningView: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (jobId) {
      loadResults();
    }
  }, [jobId]);

  const loadResults = async () => {
    try {
      const data = await apiClient.getScreeningResults(jobId!, 1, 50);
      setResults(data.results || []);
    } catch (error) {
      console.error('Failed to load results:', error);
    } finally {
      setLoading(false);
    }
  };

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'shortlist':
        return 'success';
      case 'reject':
        return 'error';
      default:
        return 'warning';
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
        Screening Results
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Resume ID</TableCell>
              <TableCell>Decision</TableCell>
              <TableCell>Score</TableCell>
              <TableCell>Matched Skills</TableCell>
              <TableCell>Processing Time</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {results.map((result) => (
              <TableRow key={result.id}>
                <TableCell>{result.resume_id.substring(0, 8)}</TableCell>
                <TableCell>
                  <Chip
                    label={result.decision}
                    color={getDecisionColor(result.decision)}
                    size="small"
                  />
                </TableCell>
                <TableCell>{result.final_score?.toFixed(1) || 'N/A'}</TableCell>
                <TableCell>{result.matched_skills?.slice(0, 3).join(', ') || 'None'}</TableCell>
                <TableCell>{result.processing_time_ms}ms</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default ScreeningView;

