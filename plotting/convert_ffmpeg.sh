resolution=1920x1080
ffmpeg -y \
  -framerate 16   \
  -s:v $resolution \
  -i gris_g900m_rcps_%04d.png \
  -c:v libx264  \
  -crf 20 \
  -pix_fmt yuv420p \
  -r 16 \
gris-g900m_rcps-hd1920.mp4
