# 🔒 SECURITY.md

## Security Policy

### Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

### Reporting a Vulnerability

We take the security of our project seriously. If you discover a security vulnerability, please follow these steps:

#### 1. **DO NOT** Create a Public Issue

Security vulnerabilities should be reported privately to avoid exploitation.

#### 2. Report Via Email

Send a detailed report to: **[your-security-email@example.com]**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

#### 3. What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: 1-3 days
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release cycle

#### 4. Disclosure Policy

- We will work with you to understand and fix the issue
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We ask that you do not publicly disclose the vulnerability until we've had a chance to fix it

---

## Security Best Practices

### For Users

#### 1. Environment Variables

**Never commit `.env` files to Git!**

```bash
# ❌ BAD - Never do this
git add .env

# ✅ GOOD - .env is in .gitignore
# Only commit .env.example with dummy values
```

#### 2. AWS Credentials

**Use IAM users, not root account:**

```bash
# ✅ GOOD - IAM user with limited permissions
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=wJal...

# ❌ BAD - Root account credentials
# NEVER use root account for applications!
```

**Rotate credentials regularly:**
```bash
# Every 90 days, create new IAM access keys
# Delete old keys after migration
```

#### 3. Database Credentials

**Use strong passwords:**

```bash
# ❌ BAD
DB_PASSWORD=password123

# ✅ GOOD
DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

**Use environment-specific databases:**
```bash
# Development
DATABASE_URL=postgresql://dev_user:dev_pass@localhost:5432/dev_db

# Production (use RDS with IAM authentication)
DATABASE_URL=postgresql://prod_user:strong_pass@rds.amazonaws.com:5432/prod_db
```

#### 4. API Keys

**Pinecone and other services:**

```bash
# Never hardcode API keys in code
# ❌ BAD
pinecone_api_key = "pcsk_..."

# ✅ GOOD - Use environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
```

#### 5. Secret Keys

**Generate strong random keys:**

```bash
# Generate secure keys
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update in .env
SECRET_KEY=<generated-key>
JWT_SECRET=<different-generated-key>
```

#### 6. Production Deployment

**Enable security features:**

```bash
# .env for production
ENVIRONMENT=production
API_DEBUG=False
CORS_ORIGINS=https://yourdomain.com
```

**Use HTTPS only:**
- Never expose API over HTTP in production
- Use AWS ALB/CloudFront with SSL certificate
- Enforce HTTPS redirects

**Enable rate limiting:**
```python
# In production, use proper rate limiting
# Example with FastAPI-Limiter
@limiter.limit("5/minute")
async def upload_resume():
    ...
```

---

## Security Features Built Into This Project

### 1. Authentication & Authorization

Currently uses JWT tokens:
```python
# Protected endpoints require JWT token
Authorization: Bearer <jwt-token>
```

**TODO for production:**
- [ ] Implement OAuth 2.0
- [ ] Add role-based access control (RBAC)
- [ ] Add API key management for external integrations

### 2. Data Protection

- **S3 Encryption**: All resume files encrypted at rest in S3
- **Database**: PostgreSQL connections use SSL in production
- **Secrets Management**: Use AWS Secrets Manager for production

### 3. Input Validation

All API endpoints validate input using Pydantic:
```python
class JobCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    # Automatic validation prevents injection attacks
```

### 4. File Upload Security

- File type validation (PDF, DOCX only)
- Size limits enforced (10MB default)
- Virus scanning recommended for production

### 5. SQL Injection Prevention

Using SQLAlchemy ORM with parameterized queries:
```python
# ✅ Safe - parameterized query
query = db.query(Resume).filter(Resume.id == resume_id)

# ❌ Never do this
query = f"SELECT * FROM resumes WHERE id = {resume_id}"
```

### 6. CORS Configuration

Configured to only allow specific origins:
```python
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## Known Security Considerations

### 1. Resume Content

**Issue**: Resumes contain PII (Personally Identifiable Information)

**Mitigations**:
- Store resumes encrypted in S3
- Implement data retention policies
- Add GDPR compliance features (data export, deletion)
- Use AWS KMS for encryption keys

### 2. AWS Credentials

**Issue**: Leaked credentials can incur costs and data breaches

**Mitigations**:
- Use IAM roles instead of access keys when possible
- Rotate credentials regularly
- Use AWS Secrets Manager in production
- Enable CloudTrail for auditing
- Set up billing alerts

### 3. Embeddings & Vector Data

**Issue**: Embeddings can sometimes leak information about training data

**Mitigations**:
- Don't share vector databases publicly
- Implement access controls on Pinecone
- Anonymize training data when possible

### 4. Model Security

**Issue**: ML models can be targets for adversarial attacks

**Mitigations**:
- Validate all inputs thoroughly
- Implement rate limiting
- Monitor for unusual patterns
- Regular model updates

---

## Compliance & Privacy

### GDPR Compliance

If handling EU resident data:

1. **Right to Access**: Allow users to download their data
2. **Right to Deletion**: Implement data deletion
3. **Data Minimization**: Only collect necessary data
4. **Consent**: Get explicit consent for data processing
5. **Data Portability**: Allow data export

### Data Handling

```python
# Example: Implement data deletion
@router.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: int):
    # Delete from database
    db.delete(resume)
    # Delete from S3
    s3.delete_object(Bucket=bucket, Key=key)
    # Delete from vector store
    pinecone.delete(ids=[resume_id])
    # Delete from backups
    # Audit log the deletion
```

---

## Security Checklist for Deployment

### Pre-Production

- [ ] All credentials in environment variables (not hardcoded)
- [ ] Strong random SECRET_KEY and JWT_SECRET generated
- [ ] AWS IAM user with minimal required permissions
- [ ] S3 buckets have proper access policies (no public access)
- [ ] Database uses strong passwords
- [ ] SSL/TLS enabled for all connections
- [ ] CORS configured for specific domains only
- [ ] Debug mode disabled (`API_DEBUG=False`)
- [ ] Error messages don't leak sensitive info
- [ ] Rate limiting implemented
- [ ] Input validation on all endpoints
- [ ] File upload size limits set
- [ ] Logging configured (but not logging secrets!)

### Production

- [ ] Use AWS Secrets Manager for credentials
- [ ] Enable AWS CloudTrail for auditing
- [ ] Set up AWS GuardDuty for threat detection
- [ ] Enable S3 versioning and object lock
- [ ] Use AWS WAF for API protection
- [ ] Implement automated security scanning
- [ ] Set up monitoring and alerts
- [ ] Regular security updates and patches
- [ ] Penetration testing completed
- [ ] Incident response plan documented
- [ ] Regular backups with encryption
- [ ] Data retention policy implemented

---

## Dependencies Security

### Monitoring

We use:
- **Dependabot**: Automated dependency updates
- **Safety**: Python package vulnerability scanner
- **npm audit**: JavaScript package security

### Manual Checks

```bash
# Python dependencies
pip install safety
safety check -r requirements.txt

# Node.js dependencies
cd frontend
npm audit

# Fix vulnerabilities
npm audit fix
```

---

## Incident Response

If a security incident occurs:

1. **Contain**: Isolate affected systems
2. **Assess**: Determine scope and impact
3. **Notify**: Inform affected users if PII was compromised
4. **Fix**: Patch the vulnerability
5. **Review**: Post-incident analysis
6. **Document**: Update security policies

---

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)
- [Python Security](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

---

## Contact

For security concerns: **[your-security-email@example.com]**

For general inquiries: Create an issue on GitHub

---

**Last Updated**: 2025
**Security Policy Version**: 1.0

