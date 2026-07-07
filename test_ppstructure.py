import os
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PPStructure

def test():
    print("Initializing PPStructure...")
    try:
        table_engine = PPStructure(show_log=True, image_orientation=False, use_gpu=False, det=False, rec=False, layout=True)
        print("PPStructure initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize PPStructure: {e}")

if __name__ == "__main__":
    test()
