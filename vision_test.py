import os
from modules import vision4_0

test_dir = os.path.join( os.getcwd(), 'test' )

def test():
    for image in os.listdir(test_dir):
        if image.endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image file: {image}")
            image_path = os.path.join(test_dir, image)
            client = vision4_0.init_client()
            #result = vision4_0.analyze_image_file(client, image)
            #vision4_0.print_result(result)
            #vision4_0.write_result(result, image)
            captions, read_lines, full_text = vision4_0.analyze_image_file(client, image_path, extract_results=True)
            with open(f"test/extracted_texts_{image}.txt", 'w', encoding='utf-8') as f:
                f.write("Captions:\n")
                for caption in captions:
                    f.write(f"{caption}\n")
                f.write("\nRead Lines:\n")
                for idx, (conf, line) in read_lines.items():
                    f.write(f"Line {idx}: '{line}' with confidence {conf}\n")
                f.write("\nFull Text:\n")
                f.write(full_text)
    
test()

#client = vision4_0.init_client()
#captions, read_text, full_text = vision4_0.analyze_image_file(client, 'testocr11.png', extract_results=True)
#print( captions, read_text, full_text )