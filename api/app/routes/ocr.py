# Copyright (C) 2021-2022, Konstantin D.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List

from fastapi import APIRouter, File, UploadFile, status
from pydantic import BaseModel

from app.schemas import OCRWordsOut
from app.vision import predictor
from doctr.io import decode_img_as_tensor

import requests

import numpy as np


router = APIRouter()


class ResultLine(object):
    def __init__(self, value, confidence):
        self.value = value
        self.confidence = confidence


class RemoteImage(BaseModel):
    image_url: str


@router.post("/", response_model=List[OCRWordsOut], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(file: UploadFile = File(...)):
    img = decode_img_as_tensor(file.file.read())
    out = predictor(np.array([img]))

    result = []
    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    result.append(ResultLine(word.value, word.confidence))

    return [OCRWordsOut(value=word.value, confidence=word.confidence)
            for word in result]


@router.post("/remote_url", response_model=List[OCRWordsOut], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(item: RemoteImage):
    """Runs docTR OCR model to analyze the input remote image"""

    response = requests.get(item.image_url)
    img = decode_img_as_tensor(response.content)
    out = predictor(np.array([img]))

    result = []
    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    result.append(ResultLine(word.value, word.confidence))

    return [OCRWordsOut(value=word.value, confidence=word.confidence)
            for word in result]
