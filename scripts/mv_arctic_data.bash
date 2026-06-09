for dir in submodules/hold/code/data/*/ ; do
    seq_name=$(basename "$dir")
    src="submodules/hold/code/data/$seq_name/build/image/"
    dst="data/$seq_name/images"

    if [ ! -d "$dst" ]; then
        if [ -d "$src" ]; then
            mkdir -p "data/$seq_name"
            mv "$src" "$dst"
            echo "Moved $seq_name"
        fi
    else
        echo "Skipping $seq_name: Destination exists"
    fi
done