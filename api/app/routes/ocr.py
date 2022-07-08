# Copyright (C) 2021-2022, Konstantin D.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Union

from fastapi import APIRouter, File, UploadFile, status
from pydantic import BaseModel

from app.schemas import OCRWordsOut
from app.vision import predictor
from doctr.io import decode_img_as_tensor, tensor_from_pil

from PIL import Image, ImageOps
import requests
from io import BytesIO
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


router = APIRouter()


class ResultLine(object):
    def __init__(self, value, confidence):
        self.value = value
        self.confidence = confidence


class RemoteImage(BaseModel):
    image_url: str


@router.post("/", response_model=List[OCRWordsOut], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(file: UploadFile = File(...)):
    print('HERE!!!')
    img = decode_img_as_tensor(file.file.read())

    out = predictor([img])

    print(out)

    result = []
    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    result.append(ResultLine(word.value, word.confidence))

    return [OCRWordsOut(value=word.value, confidence=word.confidence)
            for word in result]


@router.post("/remote_url", status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(item: RemoteImage):
    """Runs docTR OCR model to analyze the input remote image"""

    response = requests.get(item.image_url)
    img = decode_img_as_tensor(response.content)
    out = predictor([img])

    page_width, page_height = out.pages[0].dimensions

    image = Image.open(BytesIO(response.content))

    image.save('1.jpg', quality=100)
    # model = ocr_predictor(pretrained=True)
    # single_img_doc = DocumentFile.from_images(img)
    # result = model(single_img_doc)

    print(out)

    # try:
    #     exif = image._getexif()
    # except AttributeError as e:
    #     print("Could not get exif - Bad image!")
    #
    # print('exif', exif)
    #
    # (width, height) = image.size
    #
    # print('width', width)
    # print('height', height)
    #
    # # print "\n===Width x Heigh: %s x %s" % (width, height)
    # if not exif:
    #     if width > height:
    #         image = image.rotate(90)
    #         image.save('1.jpg', quality=100)
    # else:
    #     orientation_key = 274  # cf ExifTags
    #     if orientation_key in exif:
    #
    #         orientation = exif[orientation_key]
    #         rotate_values = {
    #             3: 180,
    #             6: 270,
    #             8: 90
    #         }
    #         if orientation in rotate_values:
    #             # Rotate and save the picture
    #             image = image.rotate(rotate_values[orientation])
    #             image.save('2.jpeg', quality=100, exif=str(exif))
    #     else:
    #         if width > height:
    #             image = image.rotate(90)
    #             image.save('2.jpeg', quality=100, exif=str(exif))
    #
    #
    # # fixed_image.save('1.jpeg')
    #
    # print('00000001', image)
    # print('0000000', fixed_image)

    # img = decode_img_as_tensor('1.jpeg')

    #
    # out = predictor(['/Users/imac-5k-6/Python/doctr/1.jpeg'])

    # print('out', out)
    # print('0000000', image)


    return item

    # print('0000000', image_url)
    #
    # print('!!!', image)

    # img = decode_img_as_tensor(image_url.file.read())
    # out = predictor([img])
    # result = []
    # for page in out.pages:
    #     for block in page.blocks:
    #         for line in block.lines:
    #             for word in line.words:
    #                 result.append(ResultLine(word.value, word.confidence))
    #
    # return [OCRWordsOut(value=word.value, confidence=word.confidence)
    #         for word in result]
