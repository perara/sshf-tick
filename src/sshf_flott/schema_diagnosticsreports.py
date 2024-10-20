from typing import List, Optional
from pydantic import BaseModel
import base64
import hashlib
import json
import cv2
import numpy as np
from pathlib import Path
import pickle

class Meta(BaseModel):
    versionId: str
    lastUpdated: str
    source: str

class Reference(BaseModel):
    reference: str

class Coding(BaseModel):
    system: str
    code: str
    display: Optional[str] = None

class Code(BaseModel):
    coding: List[Coding]
    text: str

class Link(BaseModel):
    reference: str

class Media(BaseModel):
    link: Link

class DiagnosticReport(BaseModel):
    resourceType: str
    id: str
    meta: Meta
    basedOn: List[Reference]
    status: str
    code: Code
    subject: Reference
    resultsInterpreter: List[Reference]
    result: List[Reference]
    media: List[Media]
    conclusionCode: List[Coding]  # Adjusted to expect a flat list of Coding objects

class DiagnosticReportList(BaseModel):
    DiagnosticReportList: List[DiagnosticReport]

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

def find_and_parse_diagnostic_reports(data_dir: Path ,cache=True) -> List[DiagnosticReport]:
    cache_dir = data_dir / 'cache'
    cache_dir.mkdir(exist_ok=True)
    report_files = [f for f in data_dir.glob('*.json') if 'diagnosticreport' in f.name.lower()]
    parsed_reports = []

    for report_file in report_files:
        file_hash = compute_file_hash(report_file)
        cache_file = cache_dir / f"{file_hash}.pkl"

        if cache and cache_file.exists():
            # Load cached data
            with cache_file.open('rb') as f:
                cached_data = pickle.load(f)
                parsed_reports.extend(cached_data)
                print(f"Loaded cached data for {report_file.name}")
        else:
            # Parse JSON
            with report_file.open('r') as file:
                try:
                    data = json.load(file)
                    report_list = data.get('DiagnosticReportList', [])
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON from file {report_file}")
                    continue

            for report in report_list:
                # Flatten the conclusionCode structure
                flattened_conclusion_code = [coding for code in report.get('conclusionCode', []) for coding in code['coding']]
                report['conclusionCode'] = flattened_conclusion_code

                report_obj = DiagnosticReport(**report)
                parsed_reports.append(report_obj)

            # Cache the parsed data
            with cache_file.open('wb') as f:
                pickle.dump(parsed_reports[-len(report_list):], f)
                print(f"Cached data for {report_file.name}")

    return parsed_reports

