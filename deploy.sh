docker build -t nulltracer-renderer /mnt/user/public/nulltracer/server/
docker stop nulltracer-renderer && docker rm nulltracer-renderer
cd /mnt/user/public/nulltracer && docker-compose up -d --build
# docker logs -f nulltracer-renderer
