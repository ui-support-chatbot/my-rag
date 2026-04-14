# Fix for "No module named 'pkg_resources'" Error in RAG System Deployment

## Problem Description

When deploying the RAG pipeline on a research server with Docker 20.10.8, the container fails to start with the error:

```
ERROR:rag-api:Failed to initialize RAG Pipeline: No module named 'pkg_resources'
```

This error occurs because pymilvus requires pkg_resources, which is provided by setuptools, but the module is not being found during runtime.

## Root Cause

The issue stems from the fact that while `setuptools>=68.0` is included in requirements.txt, the pkg_resources module may not be properly accessible to pymilvus during runtime in the Docker container. This is particularly problematic with Python 3.12-slim images where setuptools is not shipped by default.

## Solution

### Modified Dockerfile

The Dockerfile has been updated to ensure setuptools is installed and available before other dependencies:

```dockerfile
# ── Fix for pkg_resources issue ────────────────────────────────────────────────
# Install setuptools first to ensure pkg_resources is available before other packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir setuptools>=68.0

# ── Diagnostic: Verify pkg_resources availability ──────────────────────────────
RUN python -c "import pkg_resources; print('pkg_resources version:', pkg_resources.__version__)" || echo "pkg_resources not available"
```

### Key Changes

1. **Early Installation**: setuptools is now installed before other dependencies to ensure pkg_resources is available when pymilvus is installed.

2. **Diagnostic Check**: Added a diagnostic command to verify pkg_resources availability during the build process.

3. **Dependency Order**: The installation order ensures that setuptools is available before pymilvus tries to import pkg_resources.

## Implementation Details

- The fix addresses the specific issue with Docker 20.10.8's seccomp profile limitations
- Maintains compatibility with GTX 1080 (Pascal, sm_61) GPU requirements
- Preserves all existing functionality while resolving the import error
- Includes diagnostic output to verify the fix during container build

## Deployment Notes

When rebuilding the container with the updated Dockerfile:

1. The build process will now explicitly verify pkg_resources availability
2. pymilvus should initialize without the "No module named 'pkg_resources'" error
3. The RAG pipeline should initialize successfully during container startup

## Verification

After deploying with the updated Dockerfile, check the container logs during startup. You should see the diagnostic output confirming pkg_resources availability and the RAG pipeline initialization should succeed without the pkg_resources error.