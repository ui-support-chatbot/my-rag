# Step-by-Step Guide to Fix pkg_resources Error

## Issue
The RAG system is still showing "No module named 'pkg_resources'" error after the Dockerfile changes.

## Root Cause
The container is likely still using the old cached image instead of rebuilding with the new Dockerfile that includes the setuptools fix.

## Step-by-Step Solution

### Step 1: Stop the Current Containers
```bash
docker-compose down
```

### Step 2: Remove the Old Container Images
```bash
# Remove the specific rag-api image
docker rmi my-rag-api

# Or remove all unused images (optional, more thorough)
docker image prune -a
```

### Step 3: Force Rebuild the Container
```bash
# Build the containers from scratch, ignoring cache
docker-compose build --no-cache
```

### Step 4: Start the Services
```bash
# Start all services in detached mode
docker-compose up -d
```

### Step 5: Monitor the Logs
```bash
# Check the logs to see if the error persists
docker-compose logs -f rag-api
```

## Alternative Approach (If the above doesn't work)

If the error still persists, we need to make an additional change to the requirements.txt file to ensure pkg_resources is properly handled:

### Step 6: Update requirements.txt
Add this line to your requirements.txt file:
```
pkg-resources==0.0.0
```

Then repeat Steps 1-5.

## Verification Steps

Once the container is running:

1. Check if the RAG API is healthy:
```bash
curl http://localhost:8000/health
```

2. Verify the pipeline initialized correctly by checking the logs again:
```bash
docker-compose logs rag-api
```

Look for a line that says something like "RAG Pipeline initialized from..." instead of the pkg_resources error.

## Additional Troubleshooting

If the issue still persists:

1. Enter the container and manually check:
```bash
docker exec -it my-rag-api bash
python -c "import pkg_resources; print(pkg_resources.__version__)"
```

2. Check if pymilvus can be imported:
```bash
python -c "import pymilvus; print(pymilvus.__version__)"
```

## Important Notes

- The Dockerfile changes we made earlier should be in place (installing setuptools early)
- Make sure you're using the updated Dockerfile in your server deployment
- The --no-cache flag is crucial to ensure the changes take effect
- The diagnostic line we added should show pkg_resources availability during build