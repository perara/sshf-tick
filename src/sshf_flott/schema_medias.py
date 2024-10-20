import json
import base64
import hashlib
import pickle
from pathlib import Path
from typing import List, Any
from pydantic import BaseModel
import numpy as np
import cv2

# Define the Pydantic models
class Meta(BaseModel):
    versionId: str
    lastUpdated: str
    source: str

class Reference(BaseModel):
    reference: str

class Coding(BaseModel):
    system: str
    code: str
    display: str

class Modality(BaseModel):
    coding: List[Coding]

class Content(BaseModel):
    contentType: str
    data: str

class Media(BaseModel):
    resourceType: str
    id: str
    meta: Meta
    basedOn: List[Reference]
    partOf: List[Reference]
    status: str
    modality: Modality = None
    subject: Reference
    height: int
    width: int
    content: Content
    image_data: Any = None  # Field to hold OpenCV image data

class MediaList(BaseModel):
    MediaList: List[Media]

def decode_base64_image(base64_data: str) -> bytes:
    return base64.b64decode(base64_data)

def load_image_into_opencv(image_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def compute_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open('rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

def find_and_parse_medias(data_dir: Path, cache=True) -> List[Media]:
    cache_dir = data_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    media_files = [f for f in data_dir.glob('*.json') if 'media' in f.name.lower()]
    parsed_medias = []

    for media_file in media_files:
        file_hash = compute_file_hash(media_file)
        cache_file = cache_dir / f"{file_hash}.pkl"

        if cache and cache_file.exists():
            # Load cached data
            with cache_file.open('rb') as f:
                parsed_media = pickle.load(f)
                parsed_medias.append(parsed_media)
                print(f"Loaded cached data for {media_file.name}")
        else:
            # Parse JSON and decode images
            with media_file.open('r') as file:
                try:
                    data = json.load(file)
                    media_list = data.get('MediaList', [])
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON from file {media_file}")
                    continue

            for media in media_list:
                
                media_obj = Media(**media)
                image_data = decode_base64_image(media_obj.content.data)
                media_obj.image_data = load_image_into_opencv(image_data)
                parsed_medias.append(media_obj)

            # Cache the parsed data
            with cache_file.open('wb') as f:
                pickle.dump(parsed_medias[-len(media_list):], f)
                print(f"Cached data for {media_file.name}")

    return parsed_medias