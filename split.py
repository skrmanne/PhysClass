import os, sys

def write_txt_files(root_dir, split):
    with open(split + ".txt", "w") as f:
        for i in range(10):
            path = os.path.join(root_dir, split, str(i))
            for filename in os.listdir(path):
                if filename.endswith(".mp4"):
                    f.write(os.path.join(path, filename) + "\n")

# write train and test txt files
write_txt_files("data", "train")
write_txt_files("data", "test")