# Disposable maintainer reproduction; never changes the host's graphics stack.
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM debian:13-slim AS base
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    weston xwayland xauth libgl1-mesa-dri libegl1 libglx-mesa0 libglu1-mesa \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 \
    libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libdbus-1-3 libnss3 \
    libasound2 libxcomposite1 libxdamage1 libxrandr2 libxtst6 \
    python3 ca-certificates fonts-dejavu-core passwd \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/python
COPY scripts/tests/wayland/requirements.txt /opt/requirements.txt
# 3.12.16, reported in the original bundle, is unavailable through uv.
RUN uv venv --managed-python --python 3.12.13 /opt/phy210 \
    && uv pip install --python /opt/phy210/bin/python -r /opt/requirements.txt \
    && useradd --uid 1000 --create-home phytest \
    && mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix
ENV LIBGL_ALWAYS_SOFTWARE=true GALLIUM_DRIVER=llvmpipe LP_NATIVE_VECTOR_WIDTH=128 HOME=/home/phytest
WORKDIR /work
USER 1000:1000
ENTRYPOINT ["python3", "/src/scripts/tests/wayland/run.py"]

# Optional proxy comparison; this does not emulate Exo or virtgpu/virtwl.
FROM base AS sommelier
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    clang meson ninja-build cmake pkg-config libwayland-dev libgbm-dev libdrm-dev \
    libxpm-dev libpixman-1-dev libx11-xcb-dev libxcb-composite0-dev libxkbcommon-dev \
    libgtest-dev python3-jinja2 \
    && rm -rf /var/lib/apt/lists/*
ADD https://chromium.googlesource.com/chromiumos/platform2/+archive/c4c5759d85283c6cd89ad249e3806a4a75824b35/vm_tools/sommelier.tar.gz /opt/sommelier.tar.gz
RUN mkdir /opt/sommelier-src && tar -xzf /opt/sommelier.tar.gz -C /opt/sommelier-src \
    && meson setup /opt/sommelier-src/build /opt/sommelier-src -Dwith_tests=false \
    && meson compile -C /opt/sommelier-src/build
USER 1000:1000
