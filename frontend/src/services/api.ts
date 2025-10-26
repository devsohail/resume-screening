/**
 * API client for Resume Screening System
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

class APIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Jobs
  async createJob(jobData: any) {
    const response = await this.client.post('/jobs', jobData);
    return response.data;
  }

  async listJobs(params?: { status?: string; limit?: number; offset?: number }) {
    const response = await this.client.get('/jobs', { params });
    return response.data;
  }

  async getJob(jobId: string) {
    const response = await this.client.get(`/jobs/${jobId}`);
    return response.data;
  }

  async updateJob(jobId: string, jobData: any) {
    const response = await this.client.put(`/jobs/${jobId}`, jobData);
    return response.data;
  }

  async deleteJob(jobId: string) {
    await this.client.delete(`/jobs/${jobId}`);
  }

  // Screening
  async uploadResume(formData: FormData | File, jobId?: string, candidateName?: string, candidateEmail?: string) {
    // Support both FormData and individual parameters
    let data: FormData;
    
    if (formData instanceof FormData) {
      data = formData;
    } else {
      data = new FormData();
      data.append('file', formData);
      if (jobId) data.append('job_id', jobId);
      if (candidateName) data.append('candidate_name', candidateName);
      if (candidateEmail) data.append('candidate_email', candidateEmail);
    }

    const response = await this.client.post('/screening/upload-resume', data, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getScreeningResults(jobId: string, page = 1, pageSize = 20) {
    const response = await this.client.get(`/screening/results/${jobId}`, {
      params: { page, page_size: pageSize },
    });
    return response.data;
  }

  async getScreeningResult(resultId: string) {
    const response = await this.client.get(`/screening/result/${resultId}`);
    return response.data;
  }

  // Analytics
  async getAnalyticsMetrics(jobId?: string) {
    const response = await this.client.get('/analytics/metrics', {
      params: jobId ? { job_id: jobId } : {},
    });
    return response.data;
  }

  // Health
  async healthCheck() {
    const response = await this.client.get('/health');
    return response.data;
  }

  // Feedback API
  async submitFeedback(screeningResultId: string, humanDecision: string, notes: string = '') {
    const response = await this.client.post('/feedback/submit', {
      screening_result_id: screeningResultId,
      human_decision: humanDecision,
      notes: notes
    });
    return response.data;
  }

  async getTrainingStatus() {
    const response = await this.client.get('/feedback/training-status');
    return response.data;
  }

  async triggerTraining() {
    const response = await this.client.post('/feedback/trigger-training');
    return response.data;
  }

  // Bulk Screening
  async bulkScreenJob(jobId: string) {
    const response = await this.client.post(`/screening/bulk-screen-job/${jobId}`);
    return response.data;
  }

  // Get candidate feedback
  async getCandidateFeedback(resultId: string) {
    const response = await this.client.get(`/screening/feedback/${resultId}`);
    return response.data;
  }
}

export const apiClient = new APIClient();
export default apiClient;

