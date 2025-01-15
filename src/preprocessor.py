import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

class CreditCardPreprocessor:
    @staticmethod
    def extract_card_number(image):
        # Convert PIL Image to OpenCV format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(thresh)
        
        # Find credit card number using regex
        numbers = re.findall(r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}', text)
        
        if numbers:
            # Clean the number
            card_number = re.sub(r'[\s-]', '', numbers[0])
            return card_number
        return None

    @staticmethod
    def validate_card_number(card_number):
        if not card_number:
            return False
            
        # Luhn algorithm implementation
        digits = [int(d) for d in str(card_number)]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(divmod(d * 2, 10))
        return checksum % 10 == 0