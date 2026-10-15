FROM ubuntu:24.04
# Runtime libraries the packed ninfer-serve links, plus Python for the goethe command.
# libcuda is mounted from the host. It is not in this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ca-certificates curl libcurl4 \
        libavformat60 libavcodec60 libavutil58 libswscale7 \
    && rm -rf /var/lib/apt/lists/*
COPY goethe /opt/goethe/goethe
COPY nvfp4-kernels /opt/goethe/nvfp4-kernels
COPY bin /opt/goethe/bin
ENV PYTHONPATH=/opt/goethe NVFP4_HOME=/opt/goethe
WORKDIR /opt/goethe
ENTRYPOINT ["python3", "-m", "goethe"]
CMD ["serve"]
