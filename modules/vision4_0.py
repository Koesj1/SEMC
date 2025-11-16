import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def init_client():
    # Set the values of your computer vision endpoint and computer vision key
    # as environment variables:
    try:
        endpoint = os.environ["VISION_ENDPOINT"]
        key = os.environ["VISION_KEY"]
        
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this.")
        exit()

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
    return client

def analyze_image_file(client, image, extract_results=False):
    """Get a caption for the image from a file. This will be a synchronously (blocking) call."""
    try:
        with open(image, "rb") as image_data:
            result = client._analyze_from_image_data(
                image_data=image_data,
                visual_features=[VisualFeatures.DENSE_CAPTIONS, VisualFeatures.READ],
                gender_neutral_caption=True, # Optional (default is False)
            )
        if extract_results:
            return format_result(result)
        else:
            return result
    
    except Exception as e:
        print(f"Error analyzing image from file: {e}")
        return None

def analyze_image_url(client, image, extract_results=False):
    """Get a caption for the image from a URL. This will be a synchronously (blocking) call."""
    try:
        result = client.analyze_from_url(
            image_url=image,
            visual_features=[VisualFeatures.DENSE_CAPTIONS, VisualFeatures.READ],
            gender_neutral_caption=True,  # Optional (default is False)
        )
        if extract_results:
            return format_result(result)
        else:
            return result
    
    except Exception as e:
        print(f"Error analyzing image from URL: {e}")
        return None


def print_result(result):
    print("Image analysis results:")
    # Print caption results to the console
    try:
        print("Caption:")
        if result.dense_captions is not None:
         for caption in result.dense_captions['values']:
            if caption.confidence > .7:
                print( f"  '{caption.text}' with confidence {caption.confidence:.4f}")

    except Exception as e:
        print(f"Error printing caption results: {e}")

    # Print text (OCR) analysis results to the console
    try:
        print(" Read:")
        if result.read is not None:
            for line in result.read.blocks[0].lines:
                print(f"   Line: '{line.text}'")
                for word in line.words:
                    print(f"     Word: '{word.text}', Confidence {word.confidence:.4f}")

    except Exception as e:
        print(f"Error printing read results: {e}")
    
def write_result(result, image): 
    try:
        with open('extracted_texts_vision.txt', 'w', encoding='utf-8') as f:
            f.write("Image analysis results:\n")
            f.write(f"  Image: {image}\n")
            f.write("Caption:\n")
            if result.dense_captions is not None:
                for caption in result.dense_captions['values']:
                    if caption.confidence > .7:
                        f.write(f"  '{caption.text}' with confidence {caption.confidence:.4f}\n")

            f.write(" Read:\n")
            if result.read is not None:
                for line in result.read.blocks[0].lines:
                    average_confidence = sum(word.confidence for word in line.words) / len(line.words)      
                    f.write(f"   Line: '{line.text} == confidence: {average_confidence:.4f}'\n")
        print('File written: extracted_texts_vision.txt')

    except Exception as e:
        print(f"Error writing results to file: {e}")

def format_result(result, confidence=0.7):
    caption_texts = []
    read_lines = {}
    read_full_text = ""

    try:
        if result.dense_captions is not None:
            for caption in result.dense_captions['values']:
                if caption.confidence > confidence:
                    caption_texts.append(caption.text)
    except Exception as e:
        print(f"Error formatting caption results: {e}")
    try:
        if result.read is not None:
            for i in range(len(result.read.blocks[0].lines)):
                line = result.read.blocks[0].lines[i]
                average_confidence = sum(word.confidence for word in line.words) / len(line.words)
                read_lines[i] = (round(average_confidence, 2), line.text)
                read_full_text += line.text + " "
    except Exception as e:
        print(f"Error formatting read results: {e}")
    
    return caption_texts, read_lines, read_full_text



def main(image):
    client = init_client()
    result = analyze_image_file(client, image)
    write_result(result, image)
    #print_result(result)

if __name__ == "__main__":
    main(image='testocr4.jpg')
    #main(image="https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png")