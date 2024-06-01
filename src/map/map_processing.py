import fitz     # PyMuPDF
import os

# Open the PDF file
pdf_document = fitz.open("../pdfs/DnD_Map_Scans.pdf")

for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    image_list = page.get_images(full=True)
    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image_filename = f"../images/maps_page{page_num+1}_{img_index}.{image_ext}"
        with open(image_filename, "wb") as image_file:
            image_file.write(image_bytes)