# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# Functions
def retrieve_bounding_boxes(image, bounding_boxes, labels):
    """Retrieve bounding boxes

    Args:
        image (np.array): original image
        results (models.common.Detections): model outputs

    Returns:
        np.array: bounding boxes plot
    """
    # Retrieve dimensions of test image
    height, width = image.shape[:2]

    # Retrieve coordinates of UpperLeft point of each object
    coords_upleft = [box_coords[:2] for box_coords in bounding_boxes]

    # Retrieve coordinates of LowerRight point of each object
    coords_lowright = [box_coords[2:4] for box_coords in bounding_boxes]

    x1 = [int(point[0]) for point in coords_upleft]
    y1 = [int(point[1]) for point in coords_upleft]

    x2 = [int(point[0]) for point in coords_lowright]
    y2 = [int(point[1]) for point in coords_lowright]  
    
    # Set colors for classes in BGR
    color_truck = (255, 0, 0)
    color_bus = (0, 255, 0)
    color_car = (0, 0, 255)

    # Map labels
    labels = [
        color_car 
        if label == 'car' 
        else color_bus 
        if label == 'bus' 
        else color_truck 
        for label in labels
    ]

    # Initialize a black image
    image_bb = np.zeros((height, width,3), np.uint8)

    # Draw rectangles
    for points in zip(x1,y1,x2,y2,labels):
        image_bb = cv2.rectangle(image_bb, points[:2], points[2:4], points[-1], thickness=-1)    
    
    return image_bb


def order_points(points):
    """Order the points as follows: top-left, top-right, bot-right, bot-left

    Args:
    points (np.array): coordinates of rectangle

    Returns:
    np.array: ordered coordinates
    """
    # Initialzie a list of coordinates
    rect = np.zeros((4, 2), dtype = "float32")

    s = points.sum(axis = 1)

    # The top-left point will have the smallest sum
    rect[0] = points[np.argmin(s)]

    # The bottom-right point will have the largest sum
    rect[2] = points[np.argmax(s)]

    # Compute the difference between the points
    diff = np.diff(points, axis = 1)

    # The top-right point will have the smallest difference
    rect[1] = points[np.argmin(diff)]

    # The bottom-left will have the largest difference
    rect[3] = points[np.argmax(diff)]

    return rect


def four_point_transform(image, points):
    """Bird view transformation
    
    Args:
    image (np.array): original image
    points (np.array): points from which we cut

    Returns:
    tuple: warped(np.array), M(np.array), rect(np.array), dst(np.array) objects
    """
    # Obtain a consistent order of the points
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image by calculating distance between 
    # Bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image by calculating distance between 
    # The top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct the set of destination points
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ],
        dtype = "float32"
        )
    
    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Return items
    return warped, M, rect, dst


def transform_bounding_boxes(image_bounding_boxes, matrix, image_warped):
    """Transform bounding boxes to bird-eye view perspective

    Args:
        image_bounding_boxes (np.array): bounding boxes
        matrix (np.array): perspective transform matrix

    Returns:
        np.array: bounding boxes transformed to bird-eye view    
    """    
    # Retrieve dimensions of a warped image
    IMAGE_H, IMAGE_W = image_warped.shape[:2]

    # Transform bounding boxes to bird-eye view perspective
    image_warped_bb = cv2.warpPerspective(image_bounding_boxes, matrix, (IMAGE_W, IMAGE_H))

    return image_warped_bb


def bird_view_transformation_skewed(image, points, image_bounding_boxes):
    """Transform image region and bounding boxes to a bird-eye view perspective

    Args:
        image (np.array): original image
        points (np.array): key points for slicing required polygon of image
        image_bounding_boxes (np.array): bounding boxes

    Returns:
        tuple: matrix(np.array), image_warped(np.array), image_warped_bb(np.array), rect(np.array), dst(np.array) objects
    """    
    # Get image shapes
    IMAGE_H, IMAGE_W = image.shape[:2]
    
    # Get transformed image, matrix, rectangle and destination points
    image_warped, matrix, rect, dst = four_point_transform(image, points)
    
    # Transform bounding boxes to bird-eye view perspective
    image_warped_bb = transform_bounding_boxes(image_bounding_boxes, matrix, image_warped)

    return matrix, image_warped, image_warped_bb, rect, dst


def show_bird_view(image, image_bounding_boxes, image_warped, image_warped_bb, rect, dst):
    """Plot bird-eye view transformation of an input image

    Args:
        image (np.array): original image to be transformed
        points (np.array): key points for slicing required polygon of image
        image_bounding_boxes (np.array): bounding boxes plot
        image_warped (np.array): image region transformed to bird-eye view
        image_bounding_boxes (np.array): bounding boxes 
        rect (np.array): rectangle coordinates
        dst (np.array): destination coordinates
    """    

    fig, ax = plt.subplots(2, 2, figsize=(11,7))

    # Plot original image
    ax[0,0].imshow(image)

    # Plot source points
    ax[0,0].plot(rect[0][0],rect[0][1], color='red', marker='s', markersize=12)
    ax[0,0].plot(rect[1][0],rect[1][1], color='red', marker='s', markersize=12)
    ax[0,0].plot(rect[2][0],rect[2][1], color='red', marker='s', markersize=12)
    ax[0,0].plot(rect[3][0],rect[3][1], color='red', marker='s', markersize=12)

    # Plot destination points
    ax[0,0].plot(dst[0][0], dst[0][1], color='blue', marker='o', markersize=12)
    ax[0,0].plot(dst[1][0], dst[1][1], color='blue', marker='o', markersize=12)
    ax[0,0].plot(dst[2][0], dst[2][1], color='blue', marker='o', markersize=12)
    ax[0,0].plot(dst[3][0], dst[3][1], color='blue', marker='o', markersize=12)

    # Plot bird view image
    ax[0,1].imshow(cv2.cvtColor(image_warped, cv2.COLOR_BGR2RGB))

    # Plot bounding boxes
    ax[1,0].imshow(image_bounding_boxes)

    # Plot transformed bounding box image
    ax[1,1].imshow(image_warped_bb)

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')

    ax[0,0].set_title('Original Photo', fontsize=15)
    ax[0,1].set_title('Bird View', fontsize=15)
    ax[1,0].set_title('Bounding Boxes', fontsize=15)
    ax[1,1].set_title('Warped Bounding Boxes', fontsize=15)

    plt.show()    
    return 


def filter_point(point, region_coords):
    """Check if a point is inside a specified region. 

    Args:
        point (np.array): x and y coordinates of a bottom-right bounding box point
        region_coords (np.array): coordinates of the region

    Returns:
        Bool: is a point inside the specified region
    """    
    # Order the points as follows: top-left, top-right, bot-right, bot-left
    region_coords = order_points(region_coords)
    
    # Check if the point is inside the specified region
    point = Point(point)
    polygon = Polygon(region_coords)
    
    return polygon.contains(point)


def get_box_centre(box_coords):
    """Get the coordinates of the centre of a bounding box.

    Args:
        box_coords (np.array): upper-left and bottom-right points of the bounding box

    Returns:
        tuple: x and y coordinates of the centre of the bounding box
    """    
    return ((box_coords[2] + box_coords[0]) / 2 , (box_coords[3] + box_coords[1])/2)


def get_direction(point1, point2):
    """Get the direction between of a vector having two points.
    Normalize a vector to a unit vector by finding the norm using Pythagorean theorem 
    and dividing the vector by this norm.
    
    Args:
        point1 (np.array): start point of the vector
        point2 (np.array): end point of the vector

    Returns:
        list: x and y components of the arrow vector
    """    
    # Find a vector coordinates
    vector = point1 - point2

    # Get the norm 
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)

    return [vector[0] / norm, vector[1] / norm]


def get_vector_angle(point1, point2):
    """Finding an angle between x-axis and vector specified by two points.

    Args:
        point1 (np.array): start point of the vector
        point2 (np.array): end point of the vector

    Returns:
        int: angle in degrees
    """    
    x1, y1 = point1
    x2, y2 = point2
    
    # If line is vertical - return angle 90
    if x1 == x2:
        return 90
    
    # If line is horizontal - return angle 180
    if y1 == y2:
        return 180
    
    # Calculate a slope
    m = (y2-y1)/(x2-x1)
    
    # Calculate an angle in radians between the line and the x-axis
    angle_in_radians = math.atan(m)
    
    # Convert radians to degrees
    angle_in_degrees = math.degrees(angle_in_radians)

    return int(angle_in_degrees)


def plot_normalized_bounding_box(image, center_point, angle, object_class):
    """Plot normalized(with 90 degrees angles) bounding boxes.

    Args:
        image (np.array): blank image on which to plot bounding boxes
        center_point (np.array): center of the bounding box
        angle (int): angle between moving vector and x-axis
        object_class (str): predicted class of object

    Returns:
        np.array: image with normalized bounding boxes
    """    
    # Define bounding box shapes for each class(longer side is horisontal)
    box_shapes = {
        'car': [125, 90],  
        'truck': [250, 95],
        'bus': [250, 95]
    }
    # Define bounding box colors for each class
    box_color = {
        'car': (0, 0, 255),
        'truck': (255, 0, 0),
        'bus': (0, 255, 0)
    }
    
    # Retrieve dimensions of a bounding boxe depending on object class
    width_box, height_box = box_shapes[object_class]
    
    # Compute the vertices of the rectangle using cv2.boxPoints()
    rect = ((center_point[0], center_point[1]), (width_box, height_box), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the filled rectangle using cv2.fillPoly()
    cv2.fillPoly(image, [box], box_color[object_class])
    
    return image


def bird_view_transformation(image, points, bounding_boxes, labels, bounding_boxes_centres_transformed_shifted):
    """_summary_

    Args:
        image (_type_): _description_
        points (_type_): _description_
        bounding_boxes (_type_): _description_
        labels (_type_): _description_
        bounding_boxes_centres_transformed_shifted (_type_): _description_

    Returns:
        _type_: _description_
    """    
    assert type(points) == np.ndarray, f'Incorrect type of points: {type(points)}. Should be np.ndarray'   
    assert type(bounding_boxes) == np.ndarray, f'Incorrect type of bounding_boxes: {type(bounding_boxes)}. Should be np.ndarray'
    assert type(labels) == np.ndarray, f'Incorrect type of labels: {type(labels)}. Should be np.ndarray'

    
    # Retrieve bounding boxes
    image_bounding_boxes = retrieve_bounding_boxes(image, bounding_boxes, labels)
    
    # Bird View Transformation unnormalized(skewed)
    matrix, image_warped, image_warped_bb, rect, dst = bird_view_transformation_skewed(
        image,
        points,
        image_bounding_boxes
    )
    mask = [filter_point(box[2:], points) for box in bounding_boxes]

    bounding_boxes_filtered = bounding_boxes[mask]
    labels_filtered = labels[mask]
    
    bounding_boxes_centres = [
        get_box_centre(box_coords) 
        for box_coords 
        in bounding_boxes_filtered
    ]

    bounding_boxes_centres_transformed = cv2.perspectiveTransform(
        np.array([
            [
                [*centre]
                for centre
                in bounding_boxes_centres
            ]
        ]),
        matrix
    )
    
    angles = [
        get_vector_angle(coords[0], coords[1])
        for coords 
        in zip(
            bounding_boxes_centres_transformed[0],
            bounding_boxes_centres_transformed_shifted[0]
        )
    ]
    # Create a blank image to draw the rectangle on
    image_normalized = np.zeros_like(image_warped)

    for centre, angle, label in zip(bounding_boxes_centres_transformed[0], angles, labels_filtered):
        image_normalized = plot_normalized_bounding_box(image_normalized,centre,angle,label)

    return image_normalized