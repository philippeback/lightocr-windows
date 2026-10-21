from lightocr import LightOnOCR

def main():
    ocr = LightOnOCR()
    image_path = "C:/Users/Philippe Back/Pictures/Screenshots/Screenshot 2026-05-16 152443.png"  # Replace with your image path

    try:
        text = ocr.transcribe(image_path)
        print("Transcribed text:")
        print(text)
    except Exception as e:
        print(f"Error during OCR: {e}")

if __name__ == "__main__":
    main()
