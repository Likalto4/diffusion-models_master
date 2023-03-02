import csv
from typing import List
import cv2
import logging
import omidb
import os

import numpy as np
import database_config as cfg
import pandas as pd
import SimpleITK as sitk

# from tqdm import tqdm
from omidb.image import Image
from omidb.episode import Episode
from omidb.client import Client
from omidb.mark import BoundingBox
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s', level='INFO')


class stats:
    def __init__(self, N, B, M, IC):
        self.N = N
        self.M = M
        self.B = 0
        self.IC = IC
        self.image_CC = 0
        self.image_MLO = 0
        self.image_R = 0
        self.image_L = 0
        self.subtype = np.zeros(8, dtype=np.int32)

    def __repr__(self):
        return \
            f'Stats [N: {self.N}, M {self.M}, IC {self.IC},'\
            f'CC: {self.image_CC}, MLO: {self.image_MLO}, '\
            f'R: {self.image_R}, L:{self.image_L}, ' \
            f'Subtype: {np.array2string(self.subtype)} ]'


def get_breast_bbox(image: np.ndarray):
    """
    Makes a threshold of the image identifying the regions different from
    the background (0). Takes the largest (area) region (the one corresponding
    to the breast), defines the contour of this region and creates a roi
    that fits it.

    Args:
        image (np.ndarray): Breast image to be croped.
    Return:
        out_bbox (BoundingBox): Coordinates of the bounding box.
        img (np.ndarray): binary mask image of the breast.
    """

    # Threshold image with th=0 and get connected comp.
    img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
    nb_components, output, stats, _ = \
        cv2.connectedComponentsWithStats(img, connectivity=4)

    # Get the areas of each connected component
    sizes = stats[:, -1]
    # Keep the largest connected component
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    # Generate a binary mask for the breast
    img = np.zeros(output.shape, dtype=np.uint8)
    img[output == max_label] = 255

    # Obtain the contour of the breast and generate bbox.
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    out_bbox = BoundingBox(x, y, x+w, y+h)

    return out_bbox, img


def crop_breast_bbox(image):
    """
    Gets the bbox of the breast
    Args:
        image (np.ndarray): Image of the breast to crop.
    Returns:
        (np.ndarray): Cropped image.
    """
    # Crop
    bbox, _ = get_breast_bbox(image)
    return image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]


def bboxes_are_equal(bbox1: BoundingBox, bbox2: BoundingBox, area: float = 1.):
    """
    Compares the two bboxes based on taking into account their coordinates.
    If an area is passed, the comparisson checks the overlapping between
    the bboxes. If the overlapping is higher or equal than *area*, then
    the bboxes are considered the same one.
    Args:
        bbox1 (BoundingBox): Coordinates of bbox 1
        bbox2 (BoundingBox): Coordinates of bbox 2
        area (float, optional): Minimum area to consider the
            overlapping bboxes the same one. Defaults to 1.
    Returns:
        bool: True if the bboxes are the same one False otherwise
    """

    # Fast check if all coords are equal
    coords1 = [bbox1.x1, bbox1.x2, bbox1.y1, bbox1.y2]
    coords2 = [bbox2.x1, bbox2.x2, bbox2.y1, bbox2.y2]
    if np.equal(coords1, coords2).all():
        return True

    # Check overlapping areas
    if is_overlapping2D(bbox1, bbox2):
        bbox1_area = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        bbox2_area = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        min_bbox_area = bbox1_area if bbox1_area < bbox2_area else bbox2_area
        if (bbox1.x1 < bbox2.x1) and (bbox2.x1 < bbox1.x2):
            h_length = (bbox1.x2 - bbox2.x1)
        else:
            h_length = (bbox2.x2 - bbox1.x1)
        if (bbox1.y1 < bbox2.y1) and (bbox2.y1 < bbox1.y2):
            v_length = (bbox1.y2 - bbox2.y1)
        else:
            v_length = (bbox2.y2 - bbox1.y1)
        ovlp_area = h_length * v_length
        if ovlp_area/min_bbox_area >= area:
            return True
    return False


def get_random_bbox(
    breast_bbox: BoundingBox, fbn_rois: List[BoundingBox], prev_rois: List[BoundingBox],
    breast_mask: np.ndarray, normal_roi_noise: int = 500, normal_roi_size: int = 300,
    background_tolerance: float = 0.
):
    """_summary_
    From the original image it generates a random bbox that doesn't overlap with the
    lesion_roi (fbn_roi) and is different from the previously sampled ones. If this
    is not possible, None is returned and a warning is displayed.
    Args:
        bbox (BoundingBox): Breast bbox.
        fbn_rois (List[BoundingBox]): Foribiden rois, bboxes that we
            don't want the random bbox to overlap with.
        prev_rois (List[BoundingBox]): Previous rois, bboxes that have
            already been sampled.
        breast_mask (np.ndarray): Breast mask image.
        normal_roi_noise (int, optional): Distance (pixels) from the breast bbox
            center from where to extract a potential center. Defaults to 500.
        normal_roi_size (int, optional): Size of the square ROIs to extract.
            Defaults to 300.
        background_tolerance (float): Whether to allow patches in the border of
            the breast. This is the maximumm fraction of bkg allowed in the patch.
    Returns:
        bbox_random (BoundingBox or None): If a random bbox could be sampled
            then returns its coordinates, else returns None.
    """
    dims = breast_mask.shape

    # Get the center of the breast bbox.
    center_x = np.round((breast_bbox.x2 - breast_bbox.x1)/2).astype(int)
    center_y = np.round((breast_bbox.y2 - breast_bbox.y1)/2).astype(int)

    idx = 0
    ovrp = 0
    equal = 0
    np.random.seed(seed=420)
    while idx < 100:
        # Perturbe bbox center
        x = center_x + np.random.randint(-normal_roi_noise, +normal_roi_noise, )
        y = center_y + np.random.randint(-normal_roi_noise, +normal_roi_noise)

        # Discard coords out of boundary
        normal_roi_size = np.round(normal_roi_size / 2)
        y1 = np.maximum(y - normal_roi_size, 0).astype(int)
        y2 = np.minimum(y + normal_roi_size, dims[0]).astype(int)
        x1 = np.maximum(x - normal_roi_size, 0).astype(int)
        x2 = np.minimum(x + normal_roi_size, dims[1]).astype(int)
        bbox_rand = BoundingBox(x1, y1, x2, y2)

        # Check if actual bbox is different from previously generated ones
        bbox_exists = False if ((y1 == y2) or (x1 == x2)) else True
        if len(prev_rois) != 0:
            bbox_different = \
                all([~bboxes_are_equal(bbox_rand, prev_roi, 0.2) for prev_roi in prev_rois])
        else:
            bbox_different = True
        if bbox_different and bbox_exists:
            # Check if the bbox includes the origin of the breast.
            in_boundary = (x1 == 0) and (y1 == 0)
            if not in_boundary:
                # Check if bbox is overlapping with the "forbiden_rois".
                not_overlapping = \
                    all([~is_overlapping2D(bbox_rand, fbn_roi) for fbn_roi in fbn_rois])
                if not_overlapping:
                    # Check if the bbox matches the background tolerance
                    bb_mask = np.zeros(breast_mask.shape, dtype='uint8')
                    bb_mask[bbox_rand.y1:bbox_rand.y2, bbox_rand.x1:bbox_rand.x2] = 255
                    bkg_ovlp = cv2.bitwise_and(bb_mask, (255 - breast_mask))
                    if bkg_ovlp.sum()/(dims[0]*dims[1]) <= background_tolerance:
                        return bbox_rand
                else:
                    ovrp += 1
                # TODO: Check lesion rois bkg tolerance.
                # for fbn_roi in fbn_rois:
                # TODO: This could be done much faster
                #     bb_mask[fbn_roi.y1:fbn_roi.y2, fbn_roi.x1:fbn_roi.x2] = 255
                # and_image = cv2.bitwise_and(bb_mask, (255 - breast_mask))
        else:
            equal += 1
        idx += 1
    logging.warning(
        f'*** Could not get a Normal random ROI after 100 iterations. Ovlp: {ovrp}, equal: {equal}'
    )
    return None


def is_overlapping1D(bbox1, bbox2):
    return bbox1[1] >= bbox2[0] and bbox2[1] >= bbox1[0]


def is_overlapping2D(bbox1, bbox2):
    return is_overlapping1D([bbox1.x1, bbox1.x2], [bbox2.x1, bbox2.x2]) \
       and is_overlapping1D([bbox1.y1, bbox1.y2], [bbox2.y1, bbox2.y2])


def resize_mg(image_array: np.ndarray, filepath: str):
    """
    As I (Joaquin) don't know if all the images in the database have the same
    pixel spacing, this function resizes the image to the configured pixel
    spacing. This is necessary for the Radiomics approach.
    Args:
        image_array (np.ndarray)
        filepath (str)
    Returns:
        (np.ndarray)
    """
    # TODO: This is done with Sitk to Get a fast thing working but it should
    # be homogenized with the previous preprocessing steps.
    # TODO: fix bugs.

    orig_image = sitk.ReadImage(str(filepath))
    # Generate a Sitk image from the image array using the same metadata
    # as the original
    modified_image = sitk.GetImageFromArray(image_array)
    modified_image.SetOrigin(orig_image.GetOrigin())
    # modified_image.SetDirection(orig_image.GetDirection())
    # modified_image.SetMetaData(orig_image.GetMetaData())
    modified_image.SetSpacing(orig_image.GetSpacing())

    # Resample the image
    new_spacing = cfg.pixel_spacing
    orig_size = np.array(modified_image.GetSize(), dtype=np.int)
    orig_spacing = modified_image.GetSpacing()
    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]

    newimage = sitk.Resample(
        image1=modified_image,
        size=new_size,
        transform=sitk.Transform(),
        interpolator=sitk.sitkBSplineResamplerOrder2,
        outputOrigin=modified_image.GetOrigin(),
        outputSpacing=new_spacing,
        outputDirection=modified_image.GetDirection(),
        outputPixelType=modified_image.GetPixelID()
    )
    return sitk.GetArrayFromImage(newimage)


def img_preprocessing(image: Image, side: str):
    # Adjust pixels intensities if needed:
    if 'WindowWidth' in image.dcm:
        image_array = apply_voi_lut(image.dcm.pixel_array, image.dcm)
    else:
        image_array = image.dcm.pixel_array

    # Resize image to the same pixel spacing.
    if cfg.normalize_pixel_size:
        image_array = resize_mg(image_array, image.dcm_path)

    # Convert images to uint8
    if cfg.intensity_scale == 'zero_and_img_max':
        image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    else:
        if cfg.intensity_scale != 'bitwise_range':
            logging.warning('Intensity scaling method not supported, using: \'bitwise_range\'')
        image_array = \
            ((image_array - image_array.min()) / (image_array.max() - image_array.min())) * 255

    image_array = np.uint8(image_array)

    # Make all images left (or Right) sided
    if side == 'R':
        image_array = cv2.flip(image_array, 1)

    return image_array


def update_stats(copied_count, side, view, subtype, diagnos):
    if (side == "R"):
        copied_count.image_R += 1
    else:
        copied_count.image_L += 1

    if (view == "MLO" or view == "ML"):
        copied_count.image_MLO += 1
    elif (view == "CC"):
        copied_count.image_CC += 1
    if diagnos == 'M':
        aux_st = ''.join(map(str, subtype))
        copied_count.subtype[int(aux_st, 2)] += 1
        copied_count.M += 1
    else:
        copied_count.N += 1
    return copied_count


def add_line_csv(
    csv_path, img, lesion_metadata, client, subtype, episode, filename,
    side, scanner, breast_bbox, bbox_roi, extra_size, view, type_row
):
    acquisition_date = \
        img.dcm[0x0008, 0x0022].value if hasattr(img.dcm, 'Acquisition Date') else None
    patient_age_dcm = \
        img.dcm[0x0010, 0x1010].value if hasattr(img.dcm, 'PatientAge') else None
    dist_src_det = \
        img.dcm[0x0018, 0x1110].value if hasattr(img.dcm, 'DistanceSourceToDetector') else None
    dist_src_pat = \
        img.dcm[0x0018, 0x1111].value if hasattr(img.dcm, 'DistanceSourceToPatient') else None
    pixel_spacing = \
        img.dcm[0x0028, 0x0030].value if hasattr(img.dcm, 'PixelSpacing') else None
    implant = \
        img.dcm[0x0028, 0x1300].value if hasattr(img.dcm, 'BreastImplantPresent') else None
    row = [
        client, subtype, episode, img.id, filename, side, view, scanner,
        breast_bbox, bbox_roi, extra_size, acquisition_date, patient_age_dcm,
        dist_src_det, dist_src_pat, pixel_spacing, implant
    ]
    if type_row == 'lesion':
        row = row + list(lesion_metadata.values())

    with open(csv_path, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def store_rois_and_ffdm(
    lesions_subtypes: dict, lesions_metadata: dict,
    client: str, episode: str, scanner: str, side: str, view: str,
    image: omidb.image.Image, csv_path: str, copied_count: stats,
    extra_size: int = 50, n_normal_bbox: int = None
):
    output_path = os.path.dirname(csv_path)
    base_path = os.path.join(output_path, str(scanner))
    filename = f'{client}_{episode}_{image.id}_{view}'

    image_array = img_preprocessing(image, side)
    breast_bbox, breast_mask = get_breast_bbox(image_array)

    dims = image_array.shape

    # Reaad the specific configuration details
    normal_roi_noise = cfg.normal_roi_noise
    normal_roi_size = cfg.normal_roi_size
    background_tolerance = cfg.bkg_tol

    # Get lesions ROIs
    lession_rois = []
    for k, mark in enumerate(image.marks):
        lesion_id = mark.lesion_id
        if lesion_id == 'UNLINKED':
            continue
        if lesion_id not in lesions_subtypes.keys():
            logging.warning(
                f'lesion_id: {lesion_id} not in keys: {lesions_subtypes.keys()}'
            )
            continue
        bbox_roi = mark.boundingBox

        # Mirror the bbox
        if (side == 'R'):
            temp = mark.boundingBox.x2
            bbox_roi.x2 = dims[1] - mark.boundingBox.x1
            bbox_roi.x1 = dims[1] - temp

        # Adding an extra_size around the lesion.
        y1 = np.maximum(bbox_roi.y1 - extra_size, 0)
        y2 = np.minimum(bbox_roi.y2 + extra_size, dims[0])
        x1 = np.maximum(bbox_roi.x1 - extra_size, 0)
        x2 = np.minimum(bbox_roi.x2 + extra_size, dims[1])

        # Keep the rois of all the lesions in the breast
        lession_rois.append(bbox_roi)

        # Crop and save patch
        subtype = lesions_subtypes[lesion_id]
        roi_path = os.path.join(base_path, 'roi', f'st{subtype[0]}{subtype[1]}{subtype[2]}')
        os.makedirs(roi_path, exist_ok=True)
        roi_path = os.path.join(roi_path, f'{filename}_{k}.png')
        image_crop = image_array[y1:y2, x1:x2]
        cv2.imwrite(roi_path, image_crop)

        # Update csv
        lesion_metadata = lesions_metadata[lesion_id]
        add_line_csv(
            csv_path, image, lesion_metadata, client, subtype, episode, filename,
            side, scanner, breast_bbox, bbox_roi, extra_size, view, 'lesion'
        )

        # Update stats
        copied_count = update_stats(copied_count, side, view, subtype, 'M')

    # Get normal ROIs
    normal_roi_path = os.path.join(base_path, 'normal_roi')
    os.makedirs(normal_roi_path, exist_ok=True)
    normal_bboxes = []

    # Determine the desired number of normal rois
    if isinstance(n_normal_bbox, str) and (n_normal_bbox == 'same'):
        n_normal_bbox = len(lession_rois)

    for n in range(n_normal_bbox):
        # Sample the desired number of normal rois
        bbox_norm = \
            get_random_bbox(
                breast_bbox, lession_rois, normal_bboxes, breast_mask,
                normal_roi_noise, normal_roi_size, background_tolerance
            )
        if bbox_norm is None:
            break
        else:
            # Crop and save
            normal_bboxes.append(bbox_norm)
            image_crop = \
                image_array[bbox_norm.y1:bbox_norm.y2, bbox_norm.x1:bbox_norm.x2]
            normal_roi_path = os.path.join(normal_roi_path, f'{filename}_{n}.png')
            cv2.imwrite(normal_roi_path, image_crop)

            # Add ROI to csv:
            add_line_csv(
                csv_path.replace('.csv', '_normal.csv'), image, None, client, subtype, episode,
                filename, side, scanner, breast_bbox, bbox_roi, extra_size, view, 'normal'
            )

            # Update stats
            copied_count = update_stats(copied_count, side, view, subtype, 'N')

    image_path = os.path.join(base_path, 'ffdm')
    os.makedirs(image_path, exist_ok=True)
    image_path = os.path.join(image_path, f'{filename}.png')

    # Write the PNG file of only the breast region
    image_crop = \
        image_array[breast_bbox.y1:breast_bbox.y2, breast_bbox.x1:breast_bbox.x2]
    cv2.imwrite(image_path, image_crop)
    return copied_count


def add_receptor_label(
    subtype: np.ndarray, not_known: np.ndarray, receptor: str,
    value_st: str, st_code: dict
):
    """
    According to the label of the receptor it modifies de
    enconding of labels.
    Args:
        subtype (np.ndarray): Subtype code
        not_known (np.ndarray):
            Bool array identifiying the receptor without data
        receptor (str): Name of the receptor ['ER', 'PR', 'HER2']
        value_st (str): Value stored in the database
        st_code (dict): Dictionary to get the positions in the coding

    Returns:
        Updated versions of 'subtype' and 'not_known'
    """
    if value_st == 'RP':
        subtype[st_code[receptor]] = True
    elif value_st == 'RN':
        subtype[st_code[receptor]] = False
    else:
        # logging.warning(f'*** {receptor} Status NA {value_st}')
        not_known[st_code[receptor]] = True
    return subtype, not_known


def get_subtype_of_finding(value_finding: dict, st_code: dict = None):
    """
    Gets the receptor's labels from an episode information.
    Args:
        episode_data (Episode): Episode to be analysed
        st_code (dict, optional):
            Dictionary to get the positions in the subtype coding
    Returns:
        subtype (np.ndarray): Subtype enconding.
        side (str): Side in which the lesion is present.
        not_known (np.ndarray):
            Bool array identifiying the receptor without data
    """
    subtype = np.zeros(3, dtype=np.int8)
    not_known = np.zeros(3, dtype=np.int8)
    for key_st, value_st in value_finding.items():      # Subtype level
        if (key_st == 'HormoneERStatus'):               # ER Hormone Rec
            subtype, not_known = add_receptor_label(
                subtype, not_known, 'ER', value_st, st_code
            )
        if (key_st == 'HormonePRStatus'):           # PR Hormone Rec
            subtype, not_known = add_receptor_label(
                subtype, not_known, 'PR', value_st, st_code
            )
        if (key_st == 'HER2ReceptorStatus'):          # HER2 Hormone Rec
            subtype, not_known = add_receptor_label(
                subtype, not_known, 'HER2', value_st, st_code
            )
    return subtype, not_known


def get_useful_metadata_of_finding(episode_data: dict, side: str, idx):
    """
    Uses the metadata coming from the episode in the db._nbss format.
    Saves the identifier of the lesion comming from the episode metadata
    of the nbss to then be able to match it with the marks of the
    Image object in the database.
    Many "interesting" fields where scrapped.
    """

    metadata_keys = [
        'episode_type', 'patient_age', 'diag_outcome_assmt', 'uss_size_assmt',
        'mammo_size_assmt', 'diag_outcome_sry', 'opinion_lesion_sry', 'disease_grade_sry',
        'nonsurgical_treatments_sry', 'chemotherapy_sry', 'malignancy_type_sry',
        'calcification_on_spec_sry', 'whole_size_of_tumor_sry', 'disease_extent_sry',
        'benign_lesions_sry', 'size_ductal_only_sry', 'axilary_nodes_positive_sry',
        'dcis_growth_patterns_sry', 'dcis_grade_sry', 'date_sry', 'diag_outcome_cli',
        'size_mm_cli', 'lesion_pos_les', 'size_mm_screen', 'date_screen',
        'microcalcification_screen', 'microcalc_with_mass_screen', 'mass_screen'
    ]
    lesion_metadata = {}
    for key in metadata_keys:
        lesion_metadata.update({key: None})
    lesion_metadata.update({
        'episode_type': episode_data['EpisodeType'],
        'patient_age': episode_data['PatientAge'],
    })
    if 'ASSESSMENT' in episode_data.keys():
        asse = episode_data['ASSESSMENT']
        if side in asse.keys() and idx in asse[side].keys():
            asses = asse[side][idx]
            lesion_metadata.update({
                'diag_outcome_assmt': asses['DiagnosticSetOutcome'],
                'uss_size_assmt': asses['UssSizeMm'],
                'mammo_size_assmt': asses['MammoSizeMm'],
            })
    if 'SURGERY' in episode_data.keys():
        sry = episode_data['SURGERY']
        srys = sry[side][idx]
        lesion_metadata.update({
            'diag_outcome_sry': sry['diagnosticsetoutcome'],
            'opinion_lesion_sry': srys['Opinion'],
            'disease_grade_sry': srys['DiseaseGrade'],
            'nonsurgical_treatments_sry': srys['NonsurgicalTreatments'],
            'chemotherapy_sry': srys['Chemotherapy'],
            # 'malignancy_type_sry': srys['MalignancyType'],
            # 'calcification_on_spec_sry': srys['CalcificationOnSpecimen'],
            'whole_size_of_tumor_sry': srys['WholeSizeOfTumour'],
            'disease_extent_sry': srys['DiseaseExtent'],
            'benign_lesions_sry': srys['BenignLesions'],
            'size_ductal_only_sry': srys['SizeDuctalOnly'],
            'axilary_nodes_positive_sry': srys['AxillaryNodesPositive'],
            'dcis_growth_patterns_sry': srys['DcisGrowthPatterns'],
            'dcis_grade_sry': srys['DcisGrade'],
            'date_sry': srys['DatePerformed'],
        })
    if 'CLINICAL' in episode_data.keys():
        cli = episode_data['CLINICAL']
        lesion_metadata['diag_outcome_cli'] = cli['diagnosticsetoutcome']
        if side in cli.keys() and idx in cli[side].keys():
            clis = cli[side][idx]
            lesion_metadata['size_mm_cli'] = clis['SizeMm']
    if 'LESION' in episode_data.keys():
        les = episode_data['LESION']
        if side in les.keys() and idx in les[side].keys():
            less = les[side][idx]
            lesion_metadata['lesion_pos_les'] = less['LesionPosition']
    if 'SCREENING' in episode_data.keys():
        scr = episode_data['SCREENING']
        if side in scr.keys():
            scrs = scr[side]
            lesion_metadata.update({
                'size_mm_screen': scrs['SizeMm'],
                'date_screen': scrs['DateTaken'],
                'microcalcification_screen': scrs['Microcalcification'],
                'microcalc_with_mass_screen': scrs['MicrocalcWithMass'],
                'mass_screen': scrs['Mass'],
            })

    # if 'BIOPSYWIDE' in episode_data.keys():
    #     bw = episode_data['BIOPSYWIDE']
    #     bws = bw[side][idx]
    #     lesion_metadata.update({
    #         'diag_outcome_biop': bws['DiagnosticSetOutcome'],
    #         'opinion_lesion_biop': bws['Opinion'],
    #         'localization_type_biop': bws['LocalisationType'],
    #         'location_biop': bws['Location'],
    #         'date_biop': bws['DatePerformed'],
    #         # 'malignancy_type_biop': bws['MalignancyType'],
    #         # 'calcification_on_spec_biop': bws['CalcificationOnSpecimen'],
    #         # 'invasive_biop': bws['InvasivePresent'],
    #         # 'in_situ_biop': bws['InsituPresent'],
    #         # 'disease_grade_biop': bws['DiseaseGrade'],
    #         # 'dcis_grade_biop': bws['DiseaseGrade'],
    #         # 'lynph_node_biop': bws['LymphNode']
    #     })

    return lesion_metadata


def get_subtypes_and_metadata_of_lesions(
    client: Client, episode: Episode, episode_data: dict,
    sides_values: dict, side: str
):
    """
    From each lesion extract the subtype in the predefined code fashion.
    If any of the receptors wasn't evaluated or the data is missing the
    lesion is discarded, otherwise the interesting metadata and the
    subtype are stored.
    """
    st_inv_code = cfg.st_inv_code
    st_code = cfg.st_code
    lesions_subtypes = {}
    metadata_dict = {}

    for key_finding, value_finding in sides_values.items():
        subtype, not_known = get_subtype_of_finding(value_finding, st_code)
        # If any of the receptors data is missing discard the case
        if not_known.any():
            msg = [st_inv_code[bla] for bla in np.where(not_known)[0]]
            logging.warning(f'The subtypes {msg} are unkown')
            logging.warning(
                f'client: {client.id} - episode {episode.id} - side {side} ignored'
            )
        else:
            lesions_subtypes[key_finding] = subtype
            metadata_dict[key_finding] = \
                get_useful_metadata_of_finding(episode_data, side, key_finding)
    return lesions_subtypes, metadata_dict


def update_stats_patient(overall, client):
    # Save the general pathological status of the patient:
    #   M: malignant, N:Normal, CI: Interval Cancer
    # In the API, the label is given in the following order of importance:
    # CI > M > B > N
    if client.status.value == 'Interval Cancer':
        overall.IC += 1
    elif client.status.value == 'Malignant':
        overall.M += 1
    elif client.status.value == 'Benign':
        overall.B += 1
    elif client.status == 'Normal':
        overall.N += 1
    return overall


def initialize_csv(csv_path: str, type_csv: str):
    columns = [
        'client', 'subtype', 'episode', 'img_id', 'filename', 'side', 'view', 'manufacturer',
        'breast_bbox', 'bbox_roi', 'extra_size', 'acquisition_date', 'patient_age_dcm',
        'dist_src_det', 'dist_src_pat', 'pixel_spacing', 'implant'
    ]
    if type_csv == 'lesions':
        columns = columns + [
            'episode_type', 'patient_age', 'diag_outcome_assmt', 'uss_size_assmt',
            'mammo_size_assmt', 'diag_outcome_sry', 'opinion_lesion_sry', 'disease_grade_sry',
            'nonsurgical_treatments_sry', 'chemotherapy_sry', 'malignancy_type_sry',
            'calcification_on_spec_sry', 'whole_size_of_tumor_sry', 'disease_extent_sry',
            'benign_lesions_sry', 'size_ductal_only_sry', 'axilary_nodes_positive_sry',
            'dcis_growth_patterns_sry', 'dcis_grade_sry', 'date_sry', 'diag_outcome_cli',
            'size_mm_cli', 'lesion_pos_les', 'size_mm_screen', 'date_screen',
            'microcalcification_screen', 'microcalc_with_mass_screen', 'mass_screen'
        ]
    else:
        csv_path = csv_path.replace('.csv', '_normal.csv')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def fields_present_in_dcm(dcm):
    fields = ['PresentationIntentType', (0x0020, 0x0062), 'Manufacturer', (0x0018, 0x5101)]
    for field in fields:
        if field not in dcm.trait_names() and field not in list(dcm.keys()):
            return False
    return True


def generate_database(db: omidb.DB, csv_name: str, output_path: str):
    # Keep the count of the cases processed
    overall = stats(0, 0, 0, 0)
    copied_count = stats(0, 0, 0, 0)

    # Read some specific configs
    extra_size = cfg.extra_size
    n_normal_bbox = cfg.n_normal_bbox
    allowed_episode_types = cfg.allowed_episode_types
    manufact_selection = cfg.manufact_selection
    views_selection = cfg.views_selection

    # Initialize csv file
    csv_path = os.path.join(output_path, csv_name)
    initialize_csv(csv_path, 'normal')
    initialize_csv(csv_path, 'lesions')

    for client in tqdm(db, total=len(db.clients)):                                         # Client level
        overall = update_stats_patient(overall, client)
        # Access 'raw' NBSS data
        nbss_data = db._nbss(client.id)
        for episode in client.episodes:                                                    # Episode level
            # Keep only malignant cases
            if episode.has_malignant_opinions and (episode.studies is not None):
                # One episode can have more than one study
                episode_data = nbss_data.get(episode.id, {})
                for key, value in episode_data.items():
                    # Keep only episodes with surgery events
                    if key in allowed_episode_types:
                        for sides_key, sides_values in value.items():
                            # Keep only episodes with defined sides
                            if (sides_key == 'R' or sides_key == 'L'):                  # Breast level
                                side = sides_key
                                # Get the subtype and metadata for all the existing lesions
                                lesions_subtypes, lesions_metadata = \
                                    get_subtypes_and_metadata_of_lesions(
                                        client, episode, episode_data, sides_values, side
                                    )
                                if len(lesions_subtypes) == 0:
                                    continue
                                # Go trough the images in the episode and match marks with episode lesions
                                for study in episode.studies:                           # Study level
                                    for serie in study.series:                          # Series level
                                        for image in serie.images:                      # Image level
                                            # If the image exists
                                            if image.dcm_path.is_file():
                                                # If the image has lesions, is for presentation, has
                                                # the desired laterality and is in the manufacturers selection
                                                if not fields_present_in_dcm(image.dcm):
                                                    logging.warning('Dicom has mandatory fields missing')
                                                    continue
                                                for_pres = image.dcm.PresentationIntentType == 'FOR PRESENTATION'
                                                correct_side = image.dcm[(0x0020, 0x0062)].value == side
                                                manufacturer = image.dcm['Manufacturer'].value
                                                correct_manuf = \
                                                    any([
                                                        manuf.lower() in manufacturer.lower()
                                                        for manuf in manufact_selection
                                                    ])
                                                view = image.dcm[(0x0018, 0x5101)].value
                                                correct_view = view in views_selection

                                                if (image.marks and for_pres and correct_side
                                                        and correct_manuf and correct_view):
                                                    logging.debug(
                                                        f'-->> Copying case: {manufacturer}, {view}, {side}'
                                                    )
                                                    logging.debug(
                                                        f'IDs: {client.id}, {episode.id}, {serie.id}, {image.id}'
                                                    )
                                                    copied_count = store_rois_and_ffdm(
                                                        lesions_subtypes, lesions_metadata,
                                                        client.id, episode.id, manufacturer, side, view,
                                                        image, csv_path, copied_count, extra_size, n_normal_bbox
                                                    )
                                                else:
                                                    if not correct_manuf:
                                                        logging.warning(f'Manufacturer not supported: {manufacturer}')
                                                    elif not correct_view:
                                                        logging.warning(f'*** View not supported: {view}')
                                                    # elif not image.marks:
                                                    #     logging.warning(f'*** Image has no marks')
                                                    # elif not for_pres:
                                                    #     logging.warning(f'*** Image not for presentation')
                                                    # elif not correct_side:
                                                    #     logging.warning(
                                                    #         f'*** Side = {image.dcm[0x0020, 0x0062].value} not {side}'
                                                    #     )
                            else:
                                # logging.warning(f'*** Side U for client: {client.id} - episode {episode.id} ignored')
                                continue

    logging.info(f'SUMMARY copied  {copied_count}')
    logging.info(f'SUMMARY overall {overall}')


def generate_light_csv_file(csv_name: str, output_path: str):
    csv_path = os.path.join(output_path, csv_name)
    les_df = pd.read_csv(csv_path)
    pat_df = pd.DataFrame(
        columns=[
            'client_id', 'filename', 'er', 'pr', 'her2', 'type_luminal', 'type_her2',
            'type_tn', 'outcome_label', 'side', 'view', 'implant', 'patient_age'
        ]
    )
    pat_df['client_id'] = les_df['client']
    les_df['subtype'] = les_df.subtype.str.strip('[]')
    pat_df[['er', 'pr', 'her2']] = np.array([np.asarray(a) for a in les_df.subtype.str.split(' ').values])
    pat_df.replace('1', True, inplace=True)
    pat_df.replace('0', False, inplace=True)
    pat_df[['type_luminal', 'type_her2', 'type_tn']] = False
    # Following Jinwoo Son et. al 2020 Nature Scientific Reports
    pat_df.loc[(pat_df.er == True) | (pat_df.pr == True), 'type_luminal'] = True
    pat_df.loc[(pat_df.er == False) & (pat_df.pr == False) & (pat_df.her2 == True), 'type_her2'] = True
    pat_df.loc[(pat_df.er == False) & (pat_df.pr == False) & (pat_df.her2 == False), 'type_tn'] = True
    pat_df['side'] = les_df['side']
    pat_df['view'] = les_df['view']
    pat_df['manufacturer'] = les_df['manufacturer']
    pat_df['patient_age'] = les_df['patient_age_dcm']
    pat_df['outcome_label'] = 'malignant'
    pat_df['filename'] = \
        'database/selection/' + pat_df['manufacturer'] + '/roi/' + \
        [''.join(a) for a in les_df.subtype.str.split(' ')] + les_df['filename'] + '.png'

    csv_path = csv_path.replace('.csv', '_normal.csv')
    norm_df = pd.read_csv(csv_path)
    new_norm_df = pd.DataFrame(
        columns=[
            'client_id', 'filename', 'er', 'pr', 'her2', 'type_luminal', 'type_her2',
            'type_tn', 'outcome_label', 'side', 'view', 'implant', 'patient_age'
        ]
    )
    new_norm_df['client_id'] = norm_df['client']
    new_norm_df[['er', 'pr', 'her2']] = False
    new_norm_df[['type_luminal', 'type_her2', 'type_tn']] = False
    new_norm_df['side'] = norm_df['side']
    new_norm_df['view'] = norm_df['view']
    new_norm_df['manufacturer'] = norm_df['manufacturer']
    new_norm_df['patient_age'] = norm_df['patient_age_dcm']
    new_norm_df['outcome_label'] = 'normal'
    new_norm_df['filename'] = \
        'database/selection/' + new_norm_df['manufacturer'] + \
        '/normal_roi/' + norm_df['filename'] + '.png'
        
    out_df = pd.concat([pat_df, new_norm_df], ignore_index=True)
    csv_path = csv_path.replace('_normal.csv', '_light.csv')
    out_df.to_csv(csv)


def main():
    reading_path = cfg.reading_path
    output_path = cfg.output_path
    csv_name = cfg.csv_name
    clients_subset = cfg.clients_subset

    if isinstance(cfg.clients_subset, list):
        db = omidb.DB(reading_path, clients=clients_subset)
    else:
        db = omidb.DB(reading_path)

    generate_database(db, csv_name, output_path)
    generate_light_csv_file(csv_name, output_path)


if __name__ == "__main__":
    main()
