from pdf2image import convert_from_path
import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI


def pdf_to_cv2_images(pdf_path):
    return [np.array(image)[:, :, ::-1] for image in convert_from_path(pdf_path, dpi=400)]


def get_average_skew_angle(cv_images_sample):
    median_angles = []

    for sample_img in cv_images_sample:
        grayed_sample = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

        # Use binary thresholding to make text stand out
        _, thresh = cv2.threshold(grayed_sample, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect edges using Canny
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        # Use HoughLines to detect lines in the image
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            continue

        angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0]]

        # Compute the median angle of the lines
        median_angles.append(np.median(angles))

    return np.median(median_angles)


def deskew_image(cv_image, angle):
    # Rotate the image around its center
    (h, w) = cv_image.shape[:2]
    center = (w // 2, h // 2)
    r_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_image, r_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def sharpen_image(cv_image):
    return cv2.filter2D(cv_image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))


def resize_image(cv_image):
    # Get the current dimensions of the image
    height, width = cv_image.shape[:2]

    # Calculate the new dimensions, twice as large
    new_dimensions = (width * 2, height * 2)

    # Resize the image to the new dimensions
    return cv2.resize(cv_image, new_dimensions, interpolation=cv2.INTER_LINEAR)


def pre_process(image, est_skew_angle, plot=False):
    if plot:
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(image)

    # Grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Fix the skew
    image = deskew_image(image, est_skew_angle)

    # Increase image size
    image = resize_image(image)

    # Apply adaptive thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 79, 6)

    # Thinning
    image = cv2.erode(image, np.ones((3, 3), np.uint8), iterations=1)

    # This seemed to also give good results
    image = cv2.equalizeHist(image)

    if plot:
        plt.subplot(1, 2, 2)
        plt.title("Preprocessed")
        plt.imshow(image, cmap='gray')
        plt.show()

    return image


def image2text(image):
    return pytesseract.image_to_string(image, lang="eng", config="-c tessedit_char_whitelist='01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@.|()“” ' --psm 6")


def parse_pdf(filepath):
    images = pdf_to_cv2_images(filepath)

    sample_size = int(len(images) * 0.33)
    est_skew_angle = get_average_skew_angle(images[-sample_size:])
    print("Estimated Skew Angle", est_skew_angle)

    texts = []
    for image in images:
        pre_processed = pre_process(image, est_skew_angle, True)
        texts.append(image2text(pre_processed))
    return texts


def get_openai_key():
    try:
        with open('keys/openai.txt', 'r') as file:
            key = file.read().strip()  # Read the key and strip any extraneous whitespace
            return key
    except Exception:
        return ""


def extract_troubleshooting(filepath):
    texts = parse_pdf(filepath)
    client = OpenAI(api_key=get_openai_key())

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"You are an accurate and powerful LLM whose purpose is to extract troubleshooting " +
                           f"information form a user manual transcript. The transcript contains a lot of errors and " +
                           f"typos due to faulty parsing and bad image quality" +
                           f" You are expected to return a structured json response in this format:" +
                           "{'TROUBLESHOOTING': ['Problem': '...', 'Possible_Cause': '...', 'Solution': '...']}" +
                           "You are also expected to fix the typos and use the original text as accurately as possible" +
                           f" Each page of the text is separated by ------. This is the text:" +
                           f"{'------'.join(texts)}",
            }
        ],
        model="gpt-3.5-turbo-0125",
    )
    return chat_completion.choices[0].message.content