# Contributing to AI-Powered Resume Screening System

First off, thank you for considering contributing to this project! It's people like you that make this project such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Pledge

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

Before you begin, ensure you have:

1. Read the [README.md](README.md) and understand the project
2. Set up your development environment following the installation guide
3. Familiarized yourself with the codebase structure

### Development Setup

1. **Fork the repository**
   - Visit https://github.com/yourusername/resume-screening
   - Click the "Fork" button in the top-right corner

2. **Clone your fork**
```bash
git clone https://github.com/YOUR-USERNAME/resume-screening.git
cd resume-screening
```

3. **Add upstream remote**
```bash
git remote add upstream https://github.com/yourusername/resume-screening.git
```

4. **Set up development environment**
```bash
# Copy environment file
cp .env.example .env

# Start with Docker Compose
docker-compose up -d

# Or set up manually (see README.md)
```

5. **Create a branch for your changes**
```bash
git checkout -b feature/your-feature-name
```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if applicable**
- **Include your environment details** (OS, Python version, etc.)

**Bug Report Template:**

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. macOS 13.0]
 - Python Version: [e.g. 3.11.5]
 - Node Version: [e.g. 18.17.0]
 - Browser: [e.g. Chrome 115]

**Additional context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Your First Code Contribution

Unsure where to begin? You can start by looking through these issue labels:

- `good-first-issue` - Issues suitable for newcomers
- `help-wanted` - Issues that need assistance
- `bug` - Bug fixes
- `enhancement` - New features

### Pull Requests

- Fill in the pull request template
- Follow the coding standards
- Include tests for new features
- Update documentation as needed
- Ensure all tests pass

## Development Process

### Branch Naming Convention

Use descriptive branch names with prefixes:

- `feature/` - New features (e.g., `feature/bulk-upload`)
- `fix/` - Bug fixes (e.g., `fix/pdf-parsing`)
- `docs/` - Documentation (e.g., `docs/api-guide`)
- `refactor/` - Code refactoring (e.g., `refactor/database-layer`)
- `test/` - Test additions (e.g., `test/screening-endpoint`)

### Making Changes

1. **Keep changes focused**
   - One feature or fix per pull request
   - Don't combine unrelated changes

2. **Write tests**
   - Add unit tests for new functions
   - Add integration tests for new features
   - Ensure all tests pass: `pytest tests/`

3. **Update documentation**
   - Update README.md if needed
   - Update API documentation
   - Add code comments for complex logic

4. **Follow coding standards** (see below)

## Coding Standards

### Python (Backend)

We follow PEP 8 with some modifications:

```bash
# Format code with Black
black backend/

# Check linting with flake8
flake8 backend/ --max-line-length=100

# Type checking with mypy
mypy backend/

# Sort imports with isort
isort backend/
```

**Key principles:**

- Maximum line length: 100 characters
- Use type hints for function signatures
- Write docstrings for all public functions/classes
- Use meaningful variable names
- Keep functions focused and small

**Example:**

```python
from typing import List, Optional

def process_resume(
    resume_text: str,
    job_id: int,
    threshold: float = 0.7
) -> dict:
    """
    Process a resume and return screening results.
    
    Args:
        resume_text: The extracted text from resume
        job_id: ID of the job to screen against
        threshold: Minimum score threshold (0-1)
        
    Returns:
        Dictionary containing screening results
        
    Raises:
        ValueError: If job_id is invalid
    """
    # Implementation here
    pass
```

### TypeScript (Frontend)

We follow the Airbnb style guide with TypeScript:

```bash
# Lint code
npm run lint

# Format code
npm run format
```

**Key principles:**

- Use TypeScript strict mode
- Define interfaces for all data structures
- Use functional components with hooks
- Keep components small and focused
- Use meaningful component and variable names

**Example:**

```typescript
interface ResumeScreeningResult {
  candidateName: string;
  score: number;
  decision: 'accept' | 'reject' | 'review';
  explanation: string;
}

const ScreeningCard: React.FC<{ result: ResumeScreeningResult }> = ({ result }) => {
  // Component implementation
};
```

### Testing

#### Backend Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_screening.py -v
```

Write tests for:
- All new functions and methods
- Edge cases and error conditions
- API endpoints (integration tests)

**Example:**

```python
import pytest
from backend.ml.preprocessing.text_cleaner import clean_text

def test_clean_text_basic():
    """Test basic text cleaning"""
    text = "Hello   World!  "
    result = clean_text(text)
    assert result == "Hello World!"

def test_clean_text_empty():
    """Test empty string handling"""
    result = clean_text("")
    assert result == ""
```

#### Frontend Tests

```bash
# Run tests
npm test

# Run with coverage
npm test -- --coverage
```

### Documentation

- Use clear, concise language
- Include code examples
- Update API documentation
- Add inline comments for complex logic

## Commit Messages

Write clear, meaningful commit messages following the Conventional Commits specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Simple commit
feat(screening): add bulk resume upload

# With body
fix(classifier): fix accuracy calculation bug

The classifier was using incorrect normalization which led to
inflated accuracy scores. Fixed by using sklearn's normalize function.

Closes #123
```

### Best Practices

- Use present tense ("add feature" not "added feature")
- Use imperative mood ("move cursor to..." not "moves cursor to...")
- First line should be 50 characters or less
- Separate subject from body with a blank line
- Reference issues and pull requests when relevant

## Pull Request Process

### Before Submitting

1. **Sync with upstream**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run all checks**
```bash
# Backend
cd backend
black .
flake8 .
mypy .
pytest tests/

# Frontend
cd frontend
npm run lint
npm run type-check
npm test
```

3. **Update documentation**
   - README.md if needed
   - API documentation
   - Inline code comments

### Submitting

1. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

2. **Create pull request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated
- [ ] All tests passing

## Screenshots (if applicable)

## Related Issues
Closes #issue_number
```

### Review Process

1. At least one maintainer will review your PR
2. Address any requested changes
3. Once approved, a maintainer will merge your PR

### After Merge

1. **Delete your branch**
```bash
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

2. **Sync your fork**
```bash
git checkout main
git pull upstream main
git push origin main
```

## Development Tips

### Debugging

```bash
# Backend debugging
python -m pdb backend/api/main.py

# View logs
docker-compose logs -f backend
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Useful Commands

```bash
# Reset database
docker-compose down -v
docker-compose up -d
alembic upgrade head

# Clean Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Clean node_modules
cd frontend && rm -rf node_modules && npm install
```

## Questions?

- Open an issue with the `question` label
- Join discussions in GitHub Discussions
- Check existing documentation

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- Project acknowledgments

Thank you for contributing! 🎉

