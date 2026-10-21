from lightocr import LightOnOCR

def main():
    ocr = LightOnOCR()
    image_path = "sample.png"  # Replace with your image path

    try:
        text = ocr.transcribe(image_path)
        print("Transcribed text:")
        print(text)
    except Exception as e:
        print(f"Error during OCR: {e}")

if __name__ == "__main__":
    main()
