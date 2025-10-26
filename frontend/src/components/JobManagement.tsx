/**
 * Job Management Component with Create, Edit, and Resume Upload
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import { Delete, Edit, Visibility, Upload, Add, FindInPage } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../services/api';

interface Job {
  id: string;
  title: string;
  company: string;
  description: string;
  required_skills: string[];
  preferred_skills: string[];
  min_experience_years?: number;
  max_experience_years?: number;
  education_requirements?: string;
  location?: string;
  salary_range?: string;
  status: string;
  created_at: string;
}

interface JobFormData {
  title: string;
  company: string;
  description: string;
  required_skills: string;
  preferred_skills: string;
  min_experience_years: string;
  max_experience_years: string;
  education_requirements: string;
  location: string;
  salary_range: string;
}

const JobManagement: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Job Dialog State
  const [jobDialogOpen, setJobDialogOpen] = useState(false);
  const [editingJob, setEditingJob] = useState<Job | null>(null);
  const [jobFormData, setJobFormData] = useState<JobFormData>({
    title: '',
    company: '',
    description: '',
    required_skills: '',
    preferred_skills: '',
    min_experience_years: '',
    max_experience_years: '',
    education_requirements: '',
    location: '',
    salary_range: '',
  });
  
  // Resume Upload State
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedJobForUpload, setSelectedJobForUpload] = useState<string | null>(null);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [candidateName, setCandidateName] = useState('');
  const [candidateEmail, setCandidateEmail] = useState('');
  const [uploadProgress, setUploadProgress] = useState(false);
  
  // Bulk Screening State
  const [bulkScreening, setBulkScreening] = useState<{ [key: string]: boolean }>({});
  
  const navigate = useNavigate();

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.listJobs({ limit: 100 });
      setJobs(data);
    } catch (error: any) {
      console.error('Failed to load jobs:', error);
      setError('Failed to load jobs: ' + (error.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  // Job CRUD Operations
  const handleCreateJob = () => {
    setEditingJob(null);
    setJobFormData({
      title: '',
      company: '',
      description: '',
      required_skills: '',
      preferred_skills: '',
      min_experience_years: '',
      max_experience_years: '',
      education_requirements: '',
      location: '',
      salary_range: '',
    });
    setJobDialogOpen(true);
  };

  const handleEditJob = (job: Job) => {
    setEditingJob(job);
    setJobFormData({
      title: job.title,
      company: job.company,
      description: job.description,
      required_skills: job.required_skills.join(', '),
      preferred_skills: job.preferred_skills.join(', '),
      min_experience_years: job.min_experience_years?.toString() || '',
      max_experience_years: job.max_experience_years?.toString() || '',
      education_requirements: job.education_requirements || '',
      location: job.location || '',
      salary_range: job.salary_range || '',
    });
    setJobDialogOpen(true);
  };

  const handleSaveJob = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const jobData = {
        title: jobFormData.title,
        company: jobFormData.company,
        description: jobFormData.description,
        required_skills: jobFormData.required_skills
          .split(',')
          .map(s => s.trim())
          .filter(s => s),
        preferred_skills: jobFormData.preferred_skills
          .split(',')
          .map(s => s.trim())
          .filter(s => s),
        min_experience_years: jobFormData.min_experience_years
          ? parseInt(jobFormData.min_experience_years)
          : undefined,
        max_experience_years: jobFormData.max_experience_years
          ? parseInt(jobFormData.max_experience_years)
          : undefined,
        education_requirements: jobFormData.education_requirements || undefined,
        location: jobFormData.location || undefined,
        salary_range: jobFormData.salary_range || undefined,
      };

      if (editingJob) {
        await apiClient.updateJob(editingJob.id, jobData);
        setSuccess('Job updated successfully!');
      } else {
        await apiClient.createJob(jobData);
        setSuccess('Job created successfully!');
      }

      setJobDialogOpen(false);
      loadJobs();
      setTimeout(() => setSuccess(null), 3000);
    } catch (error: any) {
      console.error('Failed to save job:', error);
      setError('Failed to save job: ' + (error.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteJob = async (jobId: string) => {
    if (window.confirm('Are you sure you want to delete this job?')) {
      try {
        setLoading(true);
        await apiClient.deleteJob(jobId);
        setSuccess('Job deleted successfully!');
        loadJobs();
        setTimeout(() => setSuccess(null), 3000);
      } catch (error: any) {
        console.error('Failed to delete job:', error);
        setError('Failed to delete job: ' + (error.message || 'Unknown error'));
      } finally {
        setLoading(false);
      }
    }
  };

  // Resume Upload Operations
  const handleOpenUploadDialog = (jobId: string) => {
    setSelectedJobForUpload(jobId);
    setResumeFile(null);
    setCandidateName('');
    setCandidateEmail('');
    setUploadDialogOpen(true);
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setResumeFile(event.target.files[0]);
    }
  };

  const handleUploadResume = async () => {
    if (!resumeFile || !selectedJobForUpload) {
      setError('Please select a resume file');
      return;
    }

    try {
      setUploadProgress(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', resumeFile);
      formData.append('job_id', selectedJobForUpload);
      if (candidateName) formData.append('candidate_name', candidateName);
      if (candidateEmail) formData.append('candidate_email', candidateEmail);

      const result = await apiClient.uploadResume(formData);
      
      setSuccess(`Resume uploaded and screened! Decision: ${result.screening_result.decision}`);
      setUploadDialogOpen(false);
      setTimeout(() => setSuccess(null), 5000);
    } catch (error: any) {
      console.error('Failed to upload resume:', error);
      setError('Failed to upload resume: ' + (error.message || 'Unknown error'));
    } finally {
      setUploadProgress(false);
    }
  };

  // Bulk Screening Operation
  const handleBulkScreen = async (jobId: string, jobTitle: string) => {
    if (!confirm(`Screen all existing resumes for "${jobTitle}"?\n\nThis will screen all uploaded resumes against this job.`)) {
      return;
    }

    try {
      setBulkScreening({ ...bulkScreening, [jobId]: true });
      setError(null);
      
      const result = await apiClient.bulkScreenJob(jobId);
      
      setSuccess(
        `Bulk screening complete for "${jobTitle}"!\n` +
        `Screened: ${result.screened}, Shortlisted: ${result.shortlisted}, ` +
        `Reviewed: ${result.reviewed}, Rejected: ${result.rejected}`
      );
      setTimeout(() => setSuccess(null), 8000);
    } catch (error: any) {
      console.error('Bulk screening failed:', error);
      setError('Bulk screening failed: ' + (error.message || 'Unknown error'));
    } finally {
      setBulkScreening({ ...bulkScreening, [jobId]: false });
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Job Management</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<Add />}
          onClick={handleCreateJob}
        >
          Create New Job
        </Button>
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {/* Jobs Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Title</strong></TableCell>
              <TableCell><strong>Company</strong></TableCell>
              <TableCell><strong>Location</strong></TableCell>
              <TableCell><strong>Status</strong></TableCell>
              <TableCell><strong>Created</strong></TableCell>
              <TableCell align="right"><strong>Actions</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading && jobs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  <CircularProgress />
                </TableCell>
              </TableRow>
            ) : jobs.length === 0 ? (
              <TableRow>
                <TableCell colSpan={6} align="center">
                  No jobs found. Create your first job!
                </TableCell>
              </TableRow>
            ) : (
              jobs.map((job) => (
                <TableRow key={job.id}>
                  <TableCell>{job.title}</TableCell>
                  <TableCell>{job.company}</TableCell>
                  <TableCell>{job.location || 'N/A'}</TableCell>
                  <TableCell>
                    <Chip
                      label={job.status}
                      color={job.status === 'active' ? 'success' : 'default'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{new Date(job.created_at).toLocaleDateString()}</TableCell>
                  <TableCell align="right">
                    <IconButton
                      size="small"
                      color="secondary"
                      onClick={() => handleBulkScreen(job.id, job.title)}
                      disabled={bulkScreening[job.id]}
                      title="Screen All Resumes"
                    >
                      {bulkScreening[job.id] ? <CircularProgress size={20} /> : <FindInPage />}
                    </IconButton>
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={() => handleOpenUploadDialog(job.id)}
                      title="Upload Resume"
                    >
                      <Upload />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="info"
                      onClick={() => navigate(`/screening/${job.id}`)}
                      title="View Screening Results"
                    >
                      <Visibility />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="default"
                      onClick={() => handleEditJob(job)}
                      title="Edit Job"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleDeleteJob(job.id)}
                      title="Delete Job"
                    >
                      <Delete />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Job Create/Edit Dialog */}
      <Dialog
        open={jobDialogOpen}
        onClose={() => setJobDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>{editingJob ? 'Edit Job' : 'Create New Job'}</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              label="Job Title"
              value={jobFormData.title}
              onChange={(e) => setJobFormData({ ...jobFormData, title: e.target.value })}
              required
              fullWidth
            />
            <TextField
              label="Company"
              value={jobFormData.company}
              onChange={(e) => setJobFormData({ ...jobFormData, company: e.target.value })}
              required
              fullWidth
            />
            <TextField
              label="Description"
              value={jobFormData.description}
              onChange={(e) => setJobFormData({ ...jobFormData, description: e.target.value })}
              required
              multiline
              rows={4}
              fullWidth
            />
            <TextField
              label="Required Skills (comma-separated)"
              value={jobFormData.required_skills}
              onChange={(e) => setJobFormData({ ...jobFormData, required_skills: e.target.value })}
              placeholder="python, fastapi, aws"
              fullWidth
            />
            <TextField
              label="Preferred Skills (comma-separated)"
              value={jobFormData.preferred_skills}
              onChange={(e) => setJobFormData({ ...jobFormData, preferred_skills: e.target.value })}
              placeholder="docker, kubernetes"
              fullWidth
            />
            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                label="Min Experience (years)"
                type="number"
                value={jobFormData.min_experience_years}
                onChange={(e) =>
                  setJobFormData({ ...jobFormData, min_experience_years: e.target.value })
                }
                fullWidth
              />
              <TextField
                label="Max Experience (years)"
                type="number"
                value={jobFormData.max_experience_years}
                onChange={(e) =>
                  setJobFormData({ ...jobFormData, max_experience_years: e.target.value })
                }
                fullWidth
              />
            </Box>
            <TextField
              label="Education Requirements"
              value={jobFormData.education_requirements}
              onChange={(e) =>
                setJobFormData({ ...jobFormData, education_requirements: e.target.value })
              }
              fullWidth
            />
            <TextField
              label="Location"
              value={jobFormData.location}
              onChange={(e) => setJobFormData({ ...jobFormData, location: e.target.value })}
              fullWidth
            />
            <TextField
              label="Salary Range"
              value={jobFormData.salary_range}
              onChange={(e) => setJobFormData({ ...jobFormData, salary_range: e.target.value })}
              placeholder="$120k - $160k"
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setJobDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveJob} variant="contained" disabled={loading}>
            {loading ? <CircularProgress size={24} /> : editingJob ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Resume Upload Dialog */}
      <Dialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Upload Resume for Screening</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <Alert severity="info">
              Upload a resume (PDF, DOCX, or TXT) to automatically screen against this job.
            </Alert>
            
            <TextField
              label="Candidate Name (optional)"
              value={candidateName}
              onChange={(e) => setCandidateName(e.target.value)}
              fullWidth
            />
            
            <TextField
              label="Candidate Email (optional)"
              type="email"
              value={candidateEmail}
              onChange={(e) => setCandidateEmail(e.target.value)}
              fullWidth
            />
            
            <Button
              variant="outlined"
              component="label"
              fullWidth
            >
              {resumeFile ? resumeFile.name : 'Select Resume File'}
              <input
                type="file"
                hidden
                accept=".pdf,.docx,.txt"
                onChange={handleFileSelect}
              />
            </Button>
            
            {resumeFile && (
              <Typography variant="caption" color="textSecondary">
                File size: {(resumeFile.size / 1024).toFixed(2)} KB
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleUploadResume}
            variant="contained"
            color="primary"
            disabled={!resumeFile || uploadProgress}
          >
            {uploadProgress ? <CircularProgress size={24} /> : 'Upload & Screen'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default JobManagement;
