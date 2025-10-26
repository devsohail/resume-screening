# S3 Bucket Structure

## Single Bucket: `octa-resume-screening-data`

### Folder Structure:

```
octa-resume-screening-data/
├── resumes/                    # Resume files
│   ├── {resume_id_1}/
│   │   └── original_filename.pdf
│   ├── {resume_id_2}/
│   │   └── john_doe_resume.docx
│   └── ...
│
├── models/                     # ML Models
│   ├── classifier/
│   │   ├── v1.0.0/
│   │   │   ├── model.pt
│   │   │   └── metadata.json
│   │   └── v1.1.0/
│   │       ├── model.pt
│   │       └── metadata.json
│   └── embedder/
│       └── ...
│
└── data/                       # Training data & exports
    ├── training/
    │   ├── labeled_resumes.csv
    │   └── processed_training.pt
    ├── exports/
    │   └── screening_results_2025.csv
    └── backups/
        └── ...
```

### Path Examples:

**Resume Upload:**
```
s3://octa-resume-screening-data/resumes/a1b2c3-uuid/john_resume.pdf
```

**Model Storage:**
```
s3://octa-resume-screening-data/models/classifier/v1.0.0/model.pt
```

**Training Data:**
```
s3://octa-resume-screening-data/data/training/labeled_resumes.csv
```

### Access Pattern:

The S3Handler automatically uses folder prefixes:
- Resumes: `resumes/{resume_id}/{filename}`
- Models: `models/{model_type}/{version}/{file}`
- Data: `data/{category}/{filename}`

This keeps everything organized in a single bucket while maintaining clear separation.
