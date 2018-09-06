rsync --files-from=train_bis.txt ./ ./train/
rsync --files-from=test_bis.txt ./ ./test/
rsync --files-from=val_bis.txt ./ ./val/
