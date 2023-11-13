import numpy as np
import matplotlib.pyplot as plt
import cv2 
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class BirdViewTransformer:
    
    def __init__(self, image, points):

        assert type(points) == np.ndarray, f'Incorrect type of points: {type(points)}. Should be np.ndarray'   

        self.image = image
        self._points = points
        self._points = self.order_points()
        self.image_height, self.image_width = self.image.shape[:2]
        self.bb_centres_transformed_prev = None
        self.bb_centres_transformed_curr = None
        self.object_ids_prev = None
        self.object_ids_curr = None
        self.id_present = []

        # Define bounding box shapes for each class(longer side is vertical)
        self.box_shapes = {
            'car': [60, 80],  
            'truck': [65, 165],
            'bus': [65, 165]
        }
        # Define bounding box colors for each class
        self.box_color = {
            'car': (0, 0, 255),
            'truck': (255, 0, 0),
            'bus': (0, 255, 0)
        }
    
    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, new_points):
        if len(new_points) != 4:
            raise ValueError(f'Amount of points is {len(new_points)}. Should be 4')
        self._points = new_points
        self._points = self.order_points()

    def __repr__(self):

        # Plot original image
        plt.imshow(self.image)

        # Plot source points
        plt.plot(self.points[0][0], self.points[0][1], color='red', marker='o', markersize=12)
        plt.plot(self.points[1][0], self.points[1][1], color='red', marker='o', markersize=12)
        plt.plot(self.points[2][0], self.points[2][1], color='red', marker='o', markersize=12)
        plt.plot(self.points[3][0], self.points[3][1], color='red', marker='o', markersize=12)

        plt.title('Selected image region', fontsize=15)

        plt.show()
        return ''
    

    def show_results(self, image_normalized):
        fig, ax = plt.subplots(1, 2, figsize=(11,7))

        # Plot original image
        ax[0].imshow(self.image)

        # Plot source points
        ax[0].plot(self.points[0][0], self.points[0][1], color='red', marker='o', markersize=12)
        ax[0].plot(self.points[1][0], self.points[1][1], color='red', marker='o', markersize=12)
        ax[0].plot(self.points[2][0], self.points[2][1], color='red', marker='o', markersize=12)
        ax[0].plot(self.points[3][0], self.points[3][1], color='red', marker='o', markersize=12)

        ax[1].imshow(image_normalized)

        ax[0].set_title('Selected image region', fontsize=15)
        ax[1].set_title('Detected Bounding Boxes', fontsize=15)

        plt.show()

        return
        

    def retrieve_bounding_boxes(self, bounding_boxes, labels):
        """Retrieve bounding boxes

        Args:
            bounding_boxes (np.array): bounding boxes
            labels (np.array): predicted labels

        Returns:
            np.array: bounding boxes plot
        """

        # Retrieve coordinates of UpperLeft point of each object
        coords_upleft = [box_coords[:2] for box_coords in bounding_boxes]

        # Retrieve coordinates of LowerRight point of each object
        coords_lowright = [box_coords[2:] for box_coords in bounding_boxes]

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
            for label 
            in labels
        ]

        # Initialize a black image
        image_bb = np.zeros_like(self.image)

        # Draw rectangles
        for points in zip(x1,y1,x2,y2,labels):
            image_bb = cv2.rectangle(image_bb, points[:2], points[2:4], points[-1], thickness=-1)    
        
        return image_bb
    
    
    def order_points(self):
        """Order the points as follows: top-left, top-right, bot-right, bot-left

        Returns:
        np.array: ordered coordinates
        """
        # Initialzie a list of coordinates
        rect = np.zeros((4, 2), dtype = "float32")

        s = self.points.sum(axis = 1)

        # The top-left point will have the smallest sum
        rect[0] = self.points[np.argmin(s)]

        # The bottom-right point will have the largest sum
        rect[2] = self.points[np.argmax(s)]

        # Compute the difference between the points
        diff = np.diff(self.points, axis = 1)

        # The top-right point will have the smallest difference
        rect[1] = self.points[np.argmin(diff)]

        # The bottom-left will have the largest difference
        rect[3] = self.points[np.argmax(diff)]

        return rect
    

    def four_point_transform(self):
        """Bird view transformation
 
        Returns:
        tuple: warped(np.array), M(np.array), rect(np.array), dst(np.array) objects
        """
        # Obtain a consistent order of the points
        rect = self.order_points()
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
        warped = cv2.warpPerspective(self.image, M, (maxWidth, maxHeight))
        
        # Return items
        return warped, M, rect, dst
    

    def transform_bounding_boxes(self, image_bounding_boxes, matrix, image_warped):
        """Transform bounding boxes to bird-eye view perspective

        Args:
            image_bounding_boxes (np.array): bounding boxes
            matrix (np.array): perspective transform matrix
            image_warped (np.array): warped image

        Returns:
            np.array: bounding boxes transformed to bird-eye view    
        """    
        # Retrieve dimensions of a warped image
        IMAGE_H, IMAGE_W = image_warped.shape[:2]

        # Transform bounding boxes to bird-eye view perspective
        image_warped_bb = cv2.warpPerspective(
            image_bounding_boxes, 
            matrix, 
            (IMAGE_W, IMAGE_H)
            )

        return image_warped_bb
    

    def bird_view_transformation_skewed(self, bounding_boxes):
        """Transform image region and bounding boxes to a bird-eye view perspective

        Args:
            bounding_boxes (np.array): bounding boxes

        Returns:
            tuple: matrix(np.array), image_warped(np.array), image_warped_bb(np.array), rect(np.array), dst(np.array) objects
        """            
        # Get transformed image, matrix, rectangle and destination points
        image_warped, matrix, rect, dst = self.four_point_transform()
        
        # Transform bounding boxes to bird-eye view perspective
        image_warped_bb = self.transform_bounding_boxes(
            bounding_boxes,
            matrix, 
            image_warped
            )

        return matrix, image_warped, image_warped_bb, rect, dst
    

    def filter_point(self, point):
        """Check if a point is inside a specified region. 

        Args:
            point (np.array): x and y coordinates of a bottom-right bounding box point

        Returns:
            Bool: is a point inside the specified region
        """    
        # Order the points as follows: top-left, top-right, bot-right, bot-left
        region_coords = self.order_points()
        
        # Check if the point is inside the specified region
        point = Point(point)
        polygon = Polygon(region_coords)
        
        return polygon.contains(point)
    

    def get_box_centre(self, box_coords):
        """Get the coordinates of the centre of a bounding box.

        Args:
            box_coords (np.array): upper-left and bottom-right points of the bounding box

        Returns:
            tuple: x and y coordinates of the centre of the bounding box
        """    
        return ((box_coords[2] + box_coords[0]) / 2 , (box_coords[3] + box_coords[1])/2)
    

    def get_direction(self, point1, point2):
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
    

    def get_vector_angle(self, point1, point2):
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


    def plot_normalized_bounding_box(self, image_blank, center_point, angle, object_class):
        """Plot normalized(with 90 degrees angles) bounding boxes.

        Args:
            image_blank (np.array): blank image on which to plot bounding boxes
            center_point (np.array): center of the bounding box
            angle (int): angle between moving vector and x-axis
            object_class (str): predicted class of object

        Returns:
            np.array: image with normalized bounding boxes
        """    
        
        # Retrieve dimensions of a bounding boxe depending on object class
        height_box, width_box = self.box_shapes[object_class]
        
        # Compute the vertices of the rectangle using cv2.boxPoints()
        rect = ((center_point[0], center_point[1]), (width_box, height_box), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw the filled rectangle using cv2.fillPoly()
        cv2.fillPoly(image_blank, [box], self.box_color[object_class])
        
        return image_blank
    

    def zip_vectors(self, vectors, object_ids):
        """Zip a dictionary of the next format {object_id: object_centre}

        Args:
            vectors (np.array): transformed bounding box centres
            object_ids (np.array): object ids

        Returns:
            dict: zipped dictionary of the next format {object_id: object_centre}
        """        
        # Check if vectors and object_ids have the same length
        assert len(vectors) == len(object_ids)

        return {
            object_id: vector 
            for vector, object_id 
            in zip(vectors, object_ids)
            }
    

    def sort_vector(self, vector_prev, vector_curr, ids_prev, ids_curr):
        """Connect object centres for the common objects ids. Keep new objects.

        Args:
            vector_prev (np.array): transformed object centres for objects in a previous frame
            vector_curr (np.array): transformed object centres for objects in a current frame
            ids_prev (np.array): id of objects from the previous frame 
            ids_curr (np.array): id of objects from the current frame 

        Returns:
            dict: dictionary of the next format {object_id: [object_centre_prev, object_centre_curr]}
        """        
        # Create a dictionary of the next format {object_id: object_centre}
        zip_vector_prev = self.zip_vectors(
            vector_prev[0],
            ids_prev
            )
        zip_vector_curr = self.zip_vectors(
            vector_curr[0], 
            ids_curr
            )

        # Filter zip_vector_curr to keep only those objects which are present in zip_vector_prev
        vectors_curr_present = {
            key: value 
            for key, value 
            in zip_vector_curr.items() 
            if key in zip_vector_prev.keys()
        }
        
        # Sort vectors_curr_present to have the same order as zip_vector_prev
        sorted_vectors_present = {
            key: vectors_curr_present[key]
            for key 
            in zip_vector_prev.keys() 
            if key in vectors_curr_present.keys()
        }
        
        # Get intersection of keys in both dictionaries
        common_keys = sorted_vectors_present.keys() & zip_vector_prev.keys()  
        
        # Connect object centers for common id
        common_objects = {
            key: [sorted_vectors_present[key], zip_vector_prev[key]]
            for key 
            in common_keys
        }
        
        # Find new object ids
        curr_diff = list(set(ids_curr) - set(common_objects.keys()))

        # Add new objects
        if len(curr_diff) != 0:
            for key in curr_diff:
                common_objects[key] = [zip_vector_curr[key], zip_vector_curr[key]]
        
        return common_objects
    

    def normalize_centre(self, centre_prev:np.array, centre_curr:np.array, label):
        """Normalize box centre.

        Args:
            centre_prev (np.array): object centres in a previous frame
            centre_curr (np.array): object centres in a current frame
            label (np.array): object labels

        Returns:
            np.array: normalized centre
        """        
        # Make a copy of current centre coordinates
        centre_normalized = centre_curr.copy()
        # Retrieve box shapes of required class
        #width, height = self.box_shapes[label]
        width, height = self.box_shapes[label]
        
        # Define filters
        filter_normalize = int(height / 4)
        filter_remove = int(height / 2)
        
        # Calculate a distance by which the object moved forward
        distance = abs(centre_curr[1] - centre_prev[1])
        
        # If distance is lower than filter_normalize - no need to change coordinates
        
        # If distance is more that filter_normalize and less than filter_remove
        # then move centre back by half of passed distance
        # If objects moves forward - move back by half of passed distance
        # If objects moves back - move forward by half of passed distance
        if (distance >= filter_normalize) and (distance < filter_remove):
            if (centre_curr[1] - centre_prev[1]) >= 0:
                centre_normalized[1] -= int(distance / 2) 
            else:
                centre_normalized[1] += int(distance / 2) 
            
        # If distance is more than filter_remove - object is treated as an outlier -> remove
        if distance >= filter_remove:
            return None
        
        # Calculate a shift distance left/right
        distance_turn = centre_curr[0] - centre_prev[0]
        
        # Define a filter
        filter_normalize_turn = int(width / 5)
        
        # If shift distance is in +-filter_normalize_turn range - 
        # shift is treated as a model inaccuracy so keep older x value
        if abs(distance_turn) < filter_normalize_turn:
            centre_normalized[0] = centre_prev[0]        
        
        return centre_normalized 
    

    def analyze(self):
        """Append a list of present ids with new ones 

        Returns:
            BirdViewTransformer: BirdViewTransformer object
        """        
        bot_border, top_border = self.analyze_region
        
        for centre, obj_id in zip(np.array(self.centres_curr), self.ids_curr):
            centre_y = centre[-1]

            if (centre_y <= top_border) and (centre_y >= bot_border) and (obj_id not in self.id_present):
                self.id_present.append(obj_id)
        
        return self


    def bird_view_transformation(self, bounding_boxes, labels, object_ids):
        """Persorm Bird-Eye View Transformation

        Args:
            bounding_boxes (np.array): bounding boxes
            labels (np.array): predicted labels
            object_ids (np.array): object ids

        Returns:
            np.array: image with objects from bird-eye view perspective
        """    

        assert type(bounding_boxes) == np.ndarray, f'Incorrect type of bounding_boxes: {type(bounding_boxes)}. Should be np.ndarray'
        assert type(labels) == np.ndarray, f'Incorrect type of labels: {type(labels)}. Should be np.ndarray'
        assert len(bounding_boxes) == len(labels), f"Length mismatch between bounding_boxes and labels. Bounding_boxes: {len(bounding_boxes)}. Labels: {len(labels)}"

        # Retrieve bounding boxes
        image_bounding_boxes = self.retrieve_bounding_boxes(bounding_boxes, labels)

        # Bird View Transformation unnormalized(skewed)
        matrix, image_warped, image_warped_bb, rect, dst = self.bird_view_transformation_skewed(
            image_bounding_boxes
        )

        # Create a blank image to draw the rectangle on
        image_normalized = np.zeros_like(image_warped)

        # If there are no cars on a frame and it's an initial frame
        if len(bounding_boxes) == 0:
            return image_normalized
        
        # If there are no cars on a frame and it's not initial frame
        if (len(bounding_boxes) == 0) and self.bb_centres_transformed_prev is not None:
            self.bb_centres_transformed_curr = self.bb_centres_transformed_prev
            self.object_ids_curr = self.object_ids_prev
            return image_normalized
        
        # Build a mask of elements located in required image region
        mask = [self.filter_point(box[2:]) for box in bounding_boxes]

        # If cars are present on a frame but they are not located in specified region
        if sum(mask) == 0:
            return image_normalized
        
        # Filter attributes using mask of elements located in required image region
        bounding_boxes_filtered = bounding_boxes[mask]
        labels_filtered = labels[mask]
        object_ids_filtered = object_ids[mask]

        # Retrieve bounding boxe centres
        bounding_boxes_centres = [
            self.get_box_centre(box_coords) 
            for box_coords 
            in bounding_boxes_filtered
        ]

        # Transform centres using precomputed matrix
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

        # Update centres
        if self.bb_centres_transformed_prev is None:
            self.bb_centres_transformed_prev = bounding_boxes_centres_transformed
            self.bb_centres_transformed_curr = bounding_boxes_centres_transformed
            self.object_ids_prev = object_ids_filtered
            self.object_ids_curr = object_ids_filtered
        
        self.bb_centres_transformed_prev = self.bb_centres_transformed_curr
        self.bb_centres_transformed_curr = bounding_boxes_centres_transformed
        self.object_ids_prev = self.object_ids_curr
        self.object_ids_curr = object_ids_filtered

        # Rertieve common objects on previous and current frames
        self.common_objects = self.sort_vector(
            self.bb_centres_transformed_prev,
            self.bb_centres_transformed_curr,
            self.object_ids_prev,
            self.object_ids_curr
            )
 
        # Get labels for current objects
        zip_labels = self.zip_vectors(labels, object_ids)
        labels_curr = [
            zip_labels[key] 
            for key 
            in self.common_objects.keys()
            ]

        # Normalize centres
        normalized_current = [*(
            self.normalize_centre(centre_prev, centre_curr, label)
            for (centre_prev, centre_curr), label
            in zip(self.common_objects.values(), labels_curr)
            if self.normalize_centre(centre_prev, centre_curr, label) is not None
        )]

        # Update common_objects with normalized centres
        for value, normalized_value in zip(self.common_objects.values(), normalized_current):
            value[-1] = normalized_value

        # Angles calculation 
        self.angles_normalized = [
            self.get_vector_angle(
                centre[0],
                centre[1]
            ) 
            for centre 
            in self.common_objects.values()
        ]

        # Retrieve centres for current bounding boxes
        self.centres_curr = [
            centres[-1] 
            for centres 
            in self.common_objects.values()
            ]
        
        # Retrieve ids of current bounding boxes
        self.ids_curr = [
            id
            for id 
            in self.common_objects.keys()
            ]

        # Build an image with normalized bounding boxes
        for centre, angle, label in zip(self.centres_curr, self.angles_normalized, labels_curr):
            image_normalized = self.plot_normalized_bounding_box(
                image_normalized,
                centre,
                angle,
                label
                )
            
        # Define top and bottom borders for a count region
        if not hasattr(self, 'analyze_region'):
            self.analyze_region = [
                image_normalized.shape[0] * 0.20,
                image_normalized.shape[0] * 0.25,
                ]
        
        # Update present ids and count amount of unique ids
        self.counter = len(self.analyze().id_present)

        # Return image with normalized bounding boxes
        return image_normalized   