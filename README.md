# meteor_head_classifier

run train_probe.py:

python ssl_ckpts/train_probe.py  --data_root Path to sorted images

    Example:
    python ssl_ckpts/train_probe.py  --data_root /mnt/c/Users/ragav/meteor_head_classifier/data1/pansy/sorted_images2


run sort_unsorted.py:

python ssl_ckpts/sort_unsorted.py \
  --input_dir "Path to unsorted images" \
  --out_dir "Path to output directory" \
  --model "Path to the trained model" \
  --class_file "Path to class_names.json" \
  --batch 32 \
  --min_conf 0.55
  


    Example:
    python ssl_ckpts/sort_unsorted.py \
        --input_dir "/mnt/c/Users/ragav/meteor_head_classifier/data1/pansy/cnn_images" \
        --out_dir "/mnt/c/Users/ragav/meteor_head_classifier/data1/pansy/sorted_out" \
        --model "/mnt/c/Users/ragav/meteor_head_classifier/finetuned.keras" \
        --class_file "/mnt/c/Users/ragav/meteor_head_classifier/ssl_ckpts/class_names.json" \
        --batch 32 \
        --min_conf 0.55


run pansy_label:

python pansy_label.py \
  --predict "Path to unsorted images" \
  --model   "Path to the trained model" \
  --classes "Path to class_names.json"

    Example:
      python pansy_label.py \
      --predict "C:\Users\ragav\meteor_head_classifier\data1\pansy\cnn_images" \
      --model   "C:\Users\ragav\meteor_head_classifier\finetuned.keras" \
      --classes "C:\Users\ragav\meteor_head_classifier\ssl_ckpts\class_names.json"

