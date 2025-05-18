"""Utility functions to run Google's Pix2Struct DocVQA model.

This module exposes helpers for loading the model and running it on a PDF or
image. It is intended to be imported by the Gradio application.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
)

__all__ = [
    "load_model",
    "convert_pdf_to_image",
    "generate",
    "run_doc_vqa",
]

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME: str = "google/pix2struct-docvqa-large"


def load_model(
    model_name: str = MODEL_NAME,
) -> Tuple[Pix2StructForConditionalGeneration, Pix2StructProcessor]:
    """Load the Pix2Struct model and processor.

    Usage:
        model, processor = load_model()
    """
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to(
        DEVICE
    )
    processor = Pix2StructProcessor.from_pretrained(model_name)
    return model, processor


def convert_pdf_to_image(filename: str, page_no: int = 1) -> Image.Image:
    """Convert a specific page of a PDF to a PIL image.

    Parameters
    ----------
    filename:
        Path to the PDF file.
    page_no:
        Page number starting from ``1``.

    Usage:
        image = convert_pdf_to_image("file.pdf", page_no=1)
    """
    images = convert_from_path(filename)
    return images[page_no - 1]


def generate(
    model: Pix2StructForConditionalGeneration,
    processor: Pix2StructProcessor,
    img: Image.Image,
    questions: List[str],
) -> List[Tuple[str, str]]:
    """Run inference on an image for a list of questions.

    Usage:
        answers = generate(model, processor, image, questions)
    """
    inputs = processor(
        images=[img for _ in range(len(questions))],
        text=questions,
        return_tensors="pt",
    ).to(DEVICE)
    predictions = model.generate(**inputs, max_new_tokens=1028)
    decoded = processor.batch_decode(predictions, skip_special_tokens=True)
    return list(zip(questions, decoded))


def run_doc_vqa(
    file_path: str,
    questions: List[str],
    page_no: int = 1,
    model: Optional[Pix2StructForConditionalGeneration] = None,
    processor: Optional[Pix2StructProcessor] = None,
) -> List[Tuple[str, str]]:
    """Run document question answering on a file.

    Parameters
    ----------
    file_path:
        Path to a PDF or image file.
    questions:
        List of user questions.
    page_no:
        Page number to convert if ``file_path`` is a PDF.
    model, processor:
        Optionally provide already initialised model and processor.

    Usage:
        results = run_doc_vqa("document.pdf", ["Question?"])
    """
    if model is None or processor is None:
        model, processor = load_model()

    extension = Path(file_path).suffix.lower()
    if extension == ".pdf":
        image = convert_pdf_to_image(file_path, page_no)
    else:
        image = Image.open(file_path)

    return generate(model, processor, image, questions)
