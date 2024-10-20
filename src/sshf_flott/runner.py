# https://github.com/xmindflow/deformableLKA
# https://github.com/rezazad68/BCDU-Net

import pathlib
from typing import List

import numpy as np
import tqdm
from deepskin import wound_segmentation
from deepskin import evaluate_PWAT_score

import cv2
from pydantic import BaseModel

from sshf_flott.schema_diagnosticsreports import find_and_parse_diagnostic_reports, DiagnosticReport, Media
from sshf_flott.schema_medias import find_and_parse_medias
from sshf_flott import schema_medias

import matplotlib.pyplot as plt


class DiagnosticReportMediaMatch(BaseModel):
    diagnostic_report: DiagnosticReport
    medias: List[schema_medias.Media]


def combine_diagnostic_reports_and_medias(
    diagnostic_reports: List[DiagnosticReport],
    medias: List[schema_medias.Media]) -> List[DiagnosticReportMediaMatch]:
    fhir_combined_data = []
    for report in fhir_report_list:

        report_medias = []
        for media in report.media:

            link = media.link.reference
            media_id = link.split("/")[-1]

            # Find media with the same ID
            matched_media = None
            for fhir_media in medias:
                if fhir_media.id == media_id:
                    matched_media = fhir_media
                    break

            if matched_media:
                report_medias.append(matched_media)

        fhir_combined_data.append(DiagnosticReportMediaMatch(diagnostic_report=report, medias=report_medias))

    return fhir_combined_data


def blend_images(img, mask, alpha=0.5, color_map=cv2.COLORMAP_JET):
    color_mask = cv2.applyColorMap(mask, color_map)
    blended = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
    return blended


def increase_contrast(img, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel back with A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert back to RGB color space
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img


def process_and_save_images(fhir_combined, current_dir):
    for fhir_combined_item in tqdm.tqdm(fhir_combined):
        codes = [code.display for code in fhir_combined_item.diagnostic_report.conclusionCode]
        codes_str = ", ".join(codes)

        for media in fhir_combined_item.medias:

            # define the image filename
            images_path = current_dir / "output" / "images"
            images_path.mkdir(exist_ok=True, parents=True)
            filename = images_path / f'{media.id}.png'
            
            if not filename.exists():
                img = media.image_data
                cv2.imwrite(str(filename.absolute()), img)
    
            # load the image using OpenCV library
            img = cv2.imread(filename)


            # convert the image to RGB fmt for the display
            img = img[..., ::-1]

            segmentation = wound_segmentation(
                img=img,
                tol=0.6,
                verbose=True,
            )

            wound_mask, body_mask, bg_mask = cv2.split(segmentation)

            # display the results
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=24)
            _ = ax1.axis('off')
            ax2.imshow(segmentation)
            ax2.set_title('Segmentation mask', fontsize=24)
            _ = ax2.axis('off')
            ax3.imshow(img)
            ax3.contour(body_mask, colors='blue', linewidths=1)
            ax3.contour(wound_mask, colors='lime', linewidths=2)
            ax3.set_title('Semantic segmentation', fontsize=24)
            _ = ax3.axis('off')

            # save the image
            segmented_images_path = current_dir / "output" / "images_with_segmentation"
            segmented_images_path.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(segmented_images_path / f"{media.id}.png")
            plt.close()







def write_label_to_cases():
    for fhir_combined_item in fhir_combined:
        codes = [code.display for code in fhir_combined_item.diagnostic_report.conclusionCode]
        codes_str = ", ".join(codes)

        for media in fhir_combined_item.medias:
            img = media.image_data

            # Write codes text on the image
            cv2.putText(img, codes_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # write the image to a file
            # into images_with_codes folder
            img_with_codes_path = current_dir / "images_with_codes"
            img_with_codes_path.mkdir(exist_ok=True)
            img_path = img_with_codes_path / f"{media.id}.jpg"
            cv2.imwrite(str(img_path), img)


if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent.parent
    data_dir = current_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    fhir_medias_data = find_and_parse_medias(data_dir)[0]
    fhir_report_list = find_and_parse_diagnostic_reports(data_dir, fhir_medias_data)
    fhir_combined = combine_diagnostic_reports_and_medias(fhir_report_list, fhir_medias_data)
    # write_label_to_cases()

    process_and_save_images(fhir_combined, current_dir)
