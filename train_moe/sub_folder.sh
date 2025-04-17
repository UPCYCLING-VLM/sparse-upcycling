for dir in cultural doc+chart general ocr; do
  if [ -d "$dir" ]; then
    for sub in "$dir"/*; do
      if [ -d "$sub" ]; then
        mv "$sub" ./
      fi
    done
    rmdir "$dir"
  fi
done
