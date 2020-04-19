## Name : Abhijit Mali
##  Assignment 12 Part B

### Details of Json file :

### Example:

{"id":0,"width":400,"height":300,"file_name":"Alaskan-Malamute-1.jpg","license":1,"date_captured":""}

#### Images:

1. id : Unique id number for every image.

2. width : width of the image.

3. height : height of the image.

4. file_name : name of the file.

5. license : describe the image's license.

6. date_captured : date when image was captured. used just for reference

#### Annotations:

#### Example:

{"id":0,"image_id":"0","segmentation":[8,7,281,7,281,297,8,297],"area":79170,"bbox":[8,7,273,290],"iscrowd":0}

1. id : Annotation id.

2. image_id : Unique id of the image.

3. category_id : [0,num-categories]represents the category label. The value num-categories is reserved to represent the background category, if applicable.

4. Segmentation : (list[list(float)] or dictionary),
 
   * list : represents a list of polygons, one for each connected component of the object, each list(float) is one simple polygon in the format of [x1, x2,....xn, yn]
 
   * dict : represents the pre-pixel segmentation mask.
 
5. area : area of bounding box in pixels

6. bbox : [x, y, width, height] of bounding box

7. is_crowd : 0 or 1, explains whether this instance is labeled as coco's crowd region. Is not a crowd (meaning it’s a single object)
