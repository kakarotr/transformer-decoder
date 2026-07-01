rclone sync /Users/kakarot/Data/CPT gdrive:transformer-decoder/CPT --progress \
  --exclude "/Sengoku/PDF/**" \
  --exclude "/Sengoku/Image/**" \
  --exclude "/Sengoku/Json/**" \
  --exclude "/Sengoku/Markdown/**" \
  --exclude "/Sengoku/EPUB/**" \
  --exclude "/Sengoku/Wiki/cleaned_html/**" \
  --exclude ".DS_Store"