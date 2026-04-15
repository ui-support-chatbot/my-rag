# Essential Steps to Fix pkg_resources Error

## Updated Dockerfile
The Dockerfile has been updated to properly install setuptools and pkg_resources before other dependencies.

## Updated requirements.txt
Added pkg-resources==0.0.0 to requirements.txt to ensure proper handling of pkg_resources.

## Server Deployment Steps

1. Stop current containers:
```bash
docker-compose down
```

2. Clean all unused images:
```bash
docker image prune -a
```

3. Update your Dockerfile and requirements.txt with the fixed versions

4. Rebuild with no cache:
```bash
docker-compose build --no-cache
```

5. Start services:
```bash
docker-compose up -d
```

6. Monitor logs:
```bash
docker-compose logs -f rag-api
```

The diagnostic in the Dockerfile will now show "pkg_resources available" during the build process, confirming the fix is working.